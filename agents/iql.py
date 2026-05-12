import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID
from agents.utils import encode, softmax, uniform_policy, random_argmax, decay_epsilon


class IndependentQLearning(Agent):

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

        self.num_actions = self.game.num_actions(self.agent)
        self.learned_policy: np.ndarray = uniform_policy(self.num_actions)

        self.q_table: dict[tuple, np.ndarray] = {}
        self._s:           tuple | None = None
        self._prev_action: int   | None = None

    def _q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def update(self) -> None:
        obs = self.game.observe(self.agent)
        if obs is None:
            return
        next_state = encode(obs)

        if self._s is None or self._prev_action is None:
            self._s = next_state
            return

        reward = self.game.reward(self.agent)
        q_curr = self._q(self._s)
        q_next = self._q(next_state)

        q_curr[self._prev_action] += self.alpha * (
            reward + self.gamma * np.max(q_next) - q_curr[self._prev_action]
        )
        self._s = next_state

        self.learned_policy = softmax(q_curr)

    def action(self) -> int:
        self.update()

        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(self.num_actions))
        else:
            q = self._q(self._s)
            a = random_argmax(self.rng, q)

        self.epsilon = decay_epsilon(self.epsilon, self.epsilon_decay, self.epsilon_min)
        self._prev_action = a
        return a

    def reset(self) -> None:
        self._s = None
        self._prev_action = None

    def policy(self) -> np.ndarray:
        return self.learned_policy
