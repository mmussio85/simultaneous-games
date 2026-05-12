from itertools import product
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID
from agents.utils import encode, softmax, uniform_policy, random_argmax, decay_epsilon


class JointActionLearningAM(Agent):

    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed=None,
    ) -> None:
        super().__init__(game=game, agent=agent)
        self.rng           = np.random.default_rng(seed)
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.num_actions  = self.game.num_actions(self.agent)
        self.other_agents = [a for a in self.game.agents if a != self.agent]

        # Inicializar:
        #     Qi (s,a) = 0 para todo s ∈S,a ∈A
        #     πj (·|s) = U(Aj ) para todo j ̸= i ,s ∈S
        self.q_table: dict[tuple, dict[tuple, float]] = {}
        self.opponent_counts: dict[tuple, dict[AgentID, np.ndarray]] = {}
        
        # Estado y acción del paso anterior
        self._s:           tuple | None = None
        self._prev_action: int   | None = None

        # Política aprendida: distribución sobre las acciones propias (softmax sobre AV)
        self.learned_policy: np.ndarray = uniform_policy(self.num_actions)

    def _get_q(self, state: tuple, joint_action: tuple) -> float:
        # Retorna Q_i(s, a); si no fue visitado retorna 0.0
        return self.q_table.get(state, {}).get(joint_action, 0.0)

    def _set_q(self, state: tuple, joint_action: tuple, value: float) -> None:
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][joint_action] = value

    def _get_opponent_policy(self, state: tuple, opponent: AgentID) -> np.ndarray:
        # Retorna π_j(· | s) como distribución normalizada sobre los conteos observados.
        # Si el estado u oponente nunca fue visto, devuelve la distribución uniforme U(A_j).
        if state not in self.opponent_counts or opponent not in self.opponent_counts[state]:
            n = self.game.num_actions(opponent)
            return np.ones(n) / n
        counts = self.opponent_counts[state][opponent]
        return counts / counts.sum()

    def _update_opponent_model(self, state: tuple, obs_actions: dict) -> None:
        # Para todo j ≠ i: actualizar π_j(· | s) incrementando el conteo de la acción observada
        if state not in self.opponent_counts:
            self.opponent_counts[state] = {}
        for opponent in self.other_agents:
            if opponent not in self.opponent_counts[state]:
                self.opponent_counts[state][opponent] = np.zeros(self.game.num_actions(opponent))
            self.opponent_counts[state][opponent][obs_actions[opponent]] += 1

    def _av(self, state: tuple, my_action: int) -> float:
        # AV_i(s, a_i) = Σ_{a_{-i}} Q(s, (a_i, a_{-i})) · π_{-i}(a_{-i} | s)
        # Se itera sobre todas las combinaciones de acciones de los oponentes,
        # ponderando cada Q por la probabilidad del modelo de oponentes.
        other_agents = [a for a in self.game.agents if a != self.agent]
        opp_ranges   = [range(self.game.num_actions(a)) for a in other_agents]

        av = 0.0
        for opp_combo in product(*opp_ranges):
            # Construir acción conjunta (a_i, a_{-i}) en el orden de game.agents
            # iterador sobre la tupla de acciones de los oponentes
            # next() devuelve el siguiente elemento de la tupla
            opp_iter = iter(opp_combo)
            joint_action = tuple(
                my_action if a == self.agent else next(opp_iter)
                for a in self.game.agents
            )

            # Π_j π_j(a_j | s): probabilidad conjunta de los oponentes bajo su modelo actual
            prob = 1.0
            for k, opp in enumerate(other_agents):
                prob *= self._get_opponent_policy(state, opp)[opp_combo[k]]

            av += self._get_q(state, joint_action) * prob
        return av

    def update(self) -> None:
        obs = self.game.observe(self.agent)
        if obs is None:
            return

        # s' (estado siguiente, codificado como tupla hashable)
        next_state = encode(obs)

        # Primera llamada del episodio: inicializar s y salir sin actualizar
        if self._s is None or self._prev_action is None:
            self._s = next_state
            return

        # Observar r_i
        reward = self.game.reward(self.agent)

        if hasattr(obs, 'keys'):
            # La observación es un diccionario {agente: acción} (ej. RPS).
            # Esto significa que el juego expone la acción conjunta directamente,
            # por lo que podemos reconstruir a = (a_i, a_{-i}) y actualizar el modelo de oponentes.
            # a_i viene de _prev_action; a_{-i} viene de obs[agente_j] para cada j ≠ i.
            joint_action = tuple(
                self._prev_action if a == self.agent else obs[a]
                for a in self.game.agents
            )
             # Para todo j ≠ i: actualizar π_j(· | s) con las acciones observadas
            self._update_opponent_model(self._s, obs)
        else:
            # La observación es un array de features del entorno (ej. Foraging).
            # No contiene las acciones de los oponentes, así que no es posible
            # reconstruir la acción conjunta real ni actualizar el modelo de oponentes.
            # Se usa 0 como placeholder para poder indexar la Q-table igualmente.
            joint_action = tuple(
                self._prev_action if a == self.agent else 0
                for a in self.game.agents
            )

        # Q_i(s, a) ← Q_i(s, a) + α [ r_i + γ max_{a'} AV_i(s', a') - Q_i(s, a) ]
        q_curr      = self._get_q(self._s, joint_action)
        max_av_next = max(self._av(next_state, a) for a in range(self.num_actions))
        self._set_q(
            self._s, joint_action,
            q_curr + self.alpha * (reward + self.gamma * max_av_next - q_curr),
        )

        # s ← s'
        self._s = next_state

        # Actualizar política aprendida: softmax sobre AV_i(s, ·) en el nuevo estado
        av_vals = np.array([self._av(self._s, a) for a in range(self.num_actions)])
        self.learned_policy = softmax(av_vals)

    def action(self) -> int:
        # Procesar la transición anterior antes de elegir la nueva acción
        self.update()

        # Con prob. ε elegir a_i ~ U(A_i), si no a_i = argmax AV_i(s, ·)
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(self.num_actions))
        else:
            av_vals = np.array([self._av(self._s, a) for a in range(self.num_actions)])
            a = random_argmax(self.rng, av_vals)

        self.epsilon = decay_epsilon(self.epsilon, self.epsilon_decay, self.epsilon_min)
        self._prev_action = a
        return a

    def policy(self) -> np.ndarray:
        return self.learned_policy
