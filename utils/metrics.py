import numpy as np


def dist_to_ne(policies: np.ndarray, ne: np.ndarray) -> np.ndarray:
    """
    Distancia L1 entre la política aprendida y el Nash Equilibrium.

    Parameters
    ----------
    policies : (n_seeds, n_episodes, n_actions)
    ne       : (n_actions,)

    Returns
    -------
    (n_seeds, n_episodes)
    """
    return np.abs(policies - ne).sum(axis=-1)


def policy_to_simplex(policy: np.ndarray) -> np.ndarray:
    """
    Convierte una política de 3 acciones [p0, p1, p2] a coordenadas 2D
    del simplex (triángulo equilátero). Útil para RPS.
    """
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return policy @ v
