import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import policy_to_simplex


def smooth(x: np.ndarray, window: int = 100) -> np.ndarray:
    """Suavizado por media móvil. Devuelve array de longitud len(x) - window + 1."""
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_with_ci(
    data: np.ndarray,
    label: str,
    color: str,
    window: int = 100,
    ax=None,
    alpha_fill: float = 0.2,
):
    """
    Grafica media ± 1 std con suavizado.

    Parameters
    ----------
    data : (n_seeds, n_episodes)
    """
    ax = ax or plt.gca()
    smoothed = np.array([smooth(data[i], window) for i in range(len(data))])
    mean = smoothed.mean(axis=0)
    std  = smoothed.std(axis=0)
    x    = np.arange(len(mean))
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=alpha_fill, color=color)


def draw_simplex(ax, action_labels=("Rock", "Paper", "Scissors")):
    """Dibuja el triángulo del simplex con etiquetas."""
    tri = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]],
        fill=False, edgecolor="black", lw=2,
    )
    ax.add_patch(tri)
    offset = 0.08
    ax.text(-offset,                   -offset, action_labels[0], ha="center", fontsize=11)
    ax.text(1 + offset,                -offset, action_labels[1], ha="center", fontsize=11)
    ax.text(0.5, np.sqrt(3) / 2 + offset,       action_labels[2], ha="center", fontsize=11)
    ne_coord = policy_to_simplex(np.array([1 / 3, 1 / 3, 1 / 3]))
    ax.scatter(*ne_coord, c="red", s=120, zorder=6, label="Nash NE")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.18, np.sqrt(3) / 2 + 0.22)
    ax.axis("off")


def plot_simplex_trajectory(
    policies_over_time: np.ndarray,
    ax=None,
    color: str = "#2196F3",
    action_labels=("Rock", "Paper", "Scissors"),
    title: str = "",
):
    """
    Grafica la trayectoria de la política en el simplex.

    Parameters
    ----------
    policies_over_time : (T, 3) — política en cada episodio
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    draw_simplex(ax, action_labels)
    coords = policy_to_simplex(policies_over_time)   # (T, 2)
    n = len(coords)
    step = max(1, n // 400)
    for k in range(0, n - 1, step):
        alpha = 0.2 + 0.8 * (k / n)
        ax.plot(coords[k : k + step + 1, 0], coords[k : k + step + 1, 1],
                color=color, alpha=alpha, lw=1.2)
    ax.scatter(*coords[0],  c="green", s=80, zorder=7, label="Inicio")
    ax.scatter(*coords[-1], c="blue",  s=80, zorder=7, label="Final")
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)


def summary_table(results_dict: dict, ne: np.ndarray = None):
    """
    Imprime tabla resumen de políticas y distancia al NE.

    Parameters
    ----------
    results_dict : {matchup_name: (rewards, policies0, policies1)}
    ne           : Nash Equilibrium opcional para calcular distancia
    """
    from utils.metrics import dist_to_ne as _dist_to_ne

    header = f"{'Matchup':<22} {'Política final A0':>24}"
    if ne is not None:
        header += f"  {'Dist NE (media)':>16}  {'Dist NE (std)':>14}"
    print(header)
    print("-" * len(header))

    for name, (_, p0, _) in results_dict.items():
        pol_final = p0[:, -200:, :].mean(axis=(0, 1))
        row = f"{name:<22} {str(pol_final.round(3)):>24}"
        if ne is not None:
            d = _dist_to_ne(p0, ne)[:, -200:].mean(axis=1)
            row += f"  {d.mean():>16.4f}  {d.std():>14.4f}"
        print(row)
