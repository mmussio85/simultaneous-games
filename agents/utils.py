import numpy as np


def encode(obs) -> tuple:
    """
    Convierte una observación del juego en una tupla hashable para usarla como clave en un diccionario.
    Si la observación es un diccionario (ej. acción conjunta en RPS), extrae los valores ordenados por agente.
    Si es un array (ej. observación de grilla en Foraging), lo aplana y convierte a lista.
    """
    if hasattr(obs, 'keys'):
        return tuple(obs[a] for a in sorted(obs.keys()))
    return tuple(np.asarray(obs).flatten().tolist())


def softmax(values: np.ndarray) -> np.ndarray:
    """
    Calcula la función softmax de forma numéricamente estable.
    Convierte un vector de valores reales en una distribución de probabilidad:
    cada valor se transforma en exp(v - max(v)) y luego se normaliza.
    """
    exps = np.exp(values - np.max(values))
    return exps / exps.sum()


def uniform_policy(n: int) -> np.ndarray:
    """
    Retorna una política uniforme sobre n acciones.
    Cada acción tiene probabilidad 1/n.
    """
    return np.ones(n) / n


def random_argmax(rng: np.random.Generator, values: np.ndarray) -> int:
    """
    Retorna el índice del valor máximo de un array.
    Si hay empates, elige uno al azar para evitar sesgos hacia el primer índice.
    """
    return int(rng.choice(np.flatnonzero(values == values.max())))


def decay_epsilon(epsilon: float, epsilon_decay: float, epsilon_min: float) -> float:
    """
    Aplica el decaimiento de epsilon multiplicándolo por epsilon_decay,
    asegurando que no baje del mínimo epsilon_min.
    """
    return max(epsilon * epsilon_decay, epsilon_min)
