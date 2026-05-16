import json
import os

import numpy as np


def save_results(
    env_name: str,
    matchup_name: str,
    rewards: np.ndarray,
    policies0: np.ndarray,
    policies1: np.ndarray,
    ne=None,
    hyperparams: dict = None,
    base_dir: str = "results",
):
    """
    Guarda los resultados de un experimento en JSON.

    Estructura: results/{env_name}/{matchup_name}/results.json
    """
    out_dir = os.path.join(base_dir, env_name, matchup_name)
    os.makedirs(out_dir, exist_ok=True)

    data = {
        "environment":    env_name,
        "matchup":        matchup_name,
        "n_seeds":        int(rewards.shape[0]),
        "n_episodes":     int(rewards.shape[1]),
        "rewards":        rewards.tolist(),
        "policies_agent0": policies0.tolist(),
        "policies_agent1": policies1.tolist(),
        "nash_equilibrium": ne if ne is None else (ne.tolist() if hasattr(ne, "tolist") else ne),
        "hyperparams":    hyperparams or {},
    }

    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved → {path}")


def load_results(env_name: str, matchup_name: str, base_dir: str = "results") -> dict:
    """Carga resultados previamente guardados."""
    path = os.path.join(base_dir, env_name, matchup_name, "results.json")
    with open(path) as f:
        data = json.load(f)
    data["rewards"]          = np.array(data["rewards"])
    data["policies_agent0"]  = np.array(data["policies_agent0"])
    data["policies_agent1"]  = np.array(data["policies_agent1"])
    if data["nash_equilibrium"] is not None:
        data["nash_equilibrium"] = np.array(data["nash_equilibrium"])
    return data
