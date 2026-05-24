import numpy as np

from utils.constants import N_EPISODES, N_SEEDS
from utils.io import save_results
import matplotlib.pyplot as plt
from utils.plotting import plot_with_ci, smooth
from utils.metrics import dist_to_ne


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


def run_experiment_and_plot(
    game_factory,
    agent0_cls,
    agent1_cls,
    NE: np.ndarray,
    color: str,
    game_name: str,
    experiment_name: str,
    agent0_label: str,
    agent1_label: str,
    color1: str = '#1B5E20',
    action_label: str = 'Rock',
    action_idx: int = 0,
    n_episodes: int = N_EPISODES,
    n_seeds: int = N_SEEDS,
    agent0_kwargs: dict = None,
    agent1_kwargs: dict = None,
    show_distance_ne=False
):
    rewards, pol0, pol1 = run_experiment(
        game_factory, agent0_cls, agent1_cls, n_episodes, n_seeds, agent0_kwargs, agent1_kwargs)

    save_results(game_name, experiment_name, rewards, pol0, pol1, ne=NE.tolist())

    num_figures = 3 if show_distance_ne else 2
    fig, axes = plt.subplots(1, num_figures, figsize=(18, 5))

    # Reward
    plot_with_ci(rewards[:, :, 0], f'{agent0_label} Agent 0', color, ax=axes[0])
    axes[0].axhline(0, ls='--', c='gray', lw=1)
    axes[0].set_title(f'{agent0_label} vs {agent1_label}: Reward')
    axes[0].set_ylabel('Reward (suavizado)')
    axes[0].legend()

    if show_distance_ne: 
    # Distancia al Equilibrio de Nash
        d0 = dist_to_ne(pol0, NE)
        d1 = dist_to_ne(pol1, NE)
        plot_with_ci(d0, f'{agent0_label} (Agent 0)', color, ax=axes[1])
        plot_with_ci(d1, f'{agent1_label} (Agent 1)', color1, ax=axes[1])
        axes[1].set_title(f'{agent0_label} vs {agent1_label}: Distancia al Equilibrio de Nash')
        axes[1].set_ylabel('Distancia L1 al EN')
        axes[1].legend()

    index = 2 if show_distance_ne else 1

    # Política actual vs. promedio (seed 0)
    pol_curr = pol0[0, :, action_idx]
    pol_avg  = np.cumsum(pol0[0, :, action_idx]) / (np.arange(n_episodes) + 1)
    axes[index].plot(pol_curr, alpha=0.4, color=color, label='Política actual')
    axes[index].plot(pol_avg, color='darkgreen', lw=2, label='Política promedio acumulada')
    axes[index].axhline(NE[action_idx], ls='--', c='red', label=f'Equilibrio de Nash ({NE[action_idx]:.3f})')
    axes[index].set_title(f'{agent0_label} vs {agent1_label}: Política actual vs. promedio ({action_label})')
    axes[index].legend()

    plt.tight_layout()
    plt.savefig(f'figures/{game_name}/{experiment_name.lower()}.png', dpi=150)
    plt.show()

    # Política final
    print("Política final FP (promedio sobre seeds):")
    print(f"  Agent 0: {pol0[:, -100:, :].mean(axis=(0,1)).round(3)}")
    print(f"  Agent 1: {pol1[:, -100:, :].mean(axis=(0,1)).round(3)}")
    print(f"  Equilibrio de Nash: {NE.round(3)}")

    return rewards, pol0, pol1
