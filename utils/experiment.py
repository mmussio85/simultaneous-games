import numpy as np

from utils.constants import N_EPISODES, N_SEEDS


def run_experiment(
    game_factory,
    agent0_cls,
    agent1_cls,
    n_episodes: int = N_EPISODES,
    n_seeds: int = N_SEEDS,
    agent0_kwargs: dict = None,
    agent1_kwargs: dict = None,
):
    """
    Corre n_seeds repeticiones de un experimento en un juego de 1 paso.

    Parameters
    ----------
    game_factory  : callable sin args que devuelve una instancia nueva del juego
    agent0_cls    : clase del agente 0 (FictitiousPlay, RegretMatching, etc.)
    agent1_cls    : clase del agente 1
    n_episodes    : episodios por seed
    n_seeds       : número de seeds (0 .. n_seeds-1)
    agent0_kwargs : kwargs extra para el constructor del agente 0
    agent1_kwargs : kwargs extra para el constructor del agente 1

    Nota: no se llama a g.reset() entre episodios. Los agentes actualizan al
    inicio de action() usando la observación del step anterior; resetear el
    juego borraría esa señal y impediría aprender.

    Returns
    -------
    rewards   : (n_seeds, n_episodes, 2)  — reward de cada agente por episodio
    policies0 : (n_seeds, n_episodes, n_actions0) — política del agente 0 tras cada paso
    policies1 : (n_seeds, n_episodes, n_actions1)
    """
    agent0_kwargs = agent0_kwargs or {}
    agent1_kwargs = agent1_kwargs or {}

    all_rewards, all_pol0, all_pol1 = [], [], []

    for seed in range(n_seeds):
        g = game_factory()
        g.reset()

        a0 = agent0_cls(game=g, agent=g.agents[0], seed=seed,       **agent0_kwargs)
        a1 = agent1_cls(game=g, agent=g.agents[1], seed=seed + 100, **agent1_kwargs)
        agents = {g.agents[0]: a0, g.agents[1]: a1}

        ep_rewards, ep_pol0, ep_pol1 = [], [], []

        for _ in range(n_episodes):
            actions = {ag: agents[ag].action() for ag in g.agents}
            g.step(actions)
            ep_rewards.append([g.reward(g.agents[0]), g.reward(g.agents[1])])
            ep_pol0.append(a0.policy().copy())
            ep_pol1.append(a1.policy().copy())

        all_rewards.append(ep_rewards)
        all_pol0.append(ep_pol0)
        all_pol1.append(ep_pol1)

    return (
        np.array(all_rewards),   # (n_seeds, n_episodes, 2)
        np.array(all_pol0),      # (n_seeds, n_episodes, n_actions0)
        np.array(all_pol1),      # (n_seeds, n_episodes, n_actions1)
    )
