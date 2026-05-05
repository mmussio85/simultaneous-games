from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)

        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random
        #
        for a in self.game.agents:
            if initial is not None:
                self.count[a] = np.array(initial, dtype=float)
            else:
                self.count[a] = np.random.randint(1, 10, size=self.game.num_actions(a)).astype(float)

        self.learned_policy: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar learned_policy usando de count
        #
        for a in self.game.agents:
            self.learned_policy[a] = self.count[a] / np.sum(self.count[a])

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #
        for joint_action in product(*agents_actions):
            actions_dict = dict(zip(g.agents, joint_action))
            g.step(actions_dict)
            rewards[joint_action] = g.reward(self.agent)
            g.reset()
        return rewards

    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente.
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        agent_idx = self.game.agents.index(self.agent)
        for joint_action, reward in rewards.items():
            my_action = joint_action[agent_idx]
            prob = 1.0
            for i, a in enumerate(self.game.agents):
                if a != self.agent:
                    prob *= self.learned_policy[a][joint_action[i]]
            utility[my_action] += reward * prob
        return utility

    def bestresponse(self):
        a = None
        #
        # TODO: retornar la acción de mayor utilidad
        #
        actions = list(self.game.action_iter(self.agent))
        a = actions[np.argmax(self.get_utility())]
        return a

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()

    def policy(self):
       return self.learned_policy[self.agent]
