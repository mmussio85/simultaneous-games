import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = int(actions[self.agent])
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=np.float64)
        #
        # TODO: calcular regrets
        #
        for a_i in g.action_iter(self.agent):
            actions[self.agent] = a_i
            g.step(actions)
            u[a_i] = float(g.reward(self.agent))
            g.reset()
        r = u - u[a]
        return r

    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #
        positive_regrets = np.maximum(self.cum_regrets, 0)
        total = np.sum(positive_regrets)
        if total > 0:
            self.curr_policy = positive_regrets / total
        else:
            self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))

    def policy(self):
        return self.learned_policy
