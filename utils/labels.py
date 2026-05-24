from collections.abc import Mapping, Sequence


def agent_display_name(
    agent_id: str,
    custom_names: Mapping[str, str] | Sequence[str] | None = None,
    *,
    index: int | None = None,
) -> str:
    """
    Etiqueta legible para un id de agente (p. ej. ``agent_0`` → ``Agent 0``).

    Parameters
    ----------
    agent_id : str
        Identificador del agente en el juego (``agent_0``, ``agent_1``, …).
    custom_names : dict o secuencia, optional
        Nombres personalizados. Como dict: ``{agent_id: etiqueta}``.
        Como lista/tupla: etiquetas por posición (``index`` requerido).
    index : int, optional
        Posición del agente cuando ``custom_names`` es una secuencia.

    Examples
    --------
    >>> agent_display_name("agent_0")
    'Agent 0'
    >>> agent_display_name("agent_1", {"agent_1": "Regret Matching"})
    'Regret Matching'
    >>> agent_display_name("agent_2", ["FP", "FP", "RM"], index=2)
    'RM'
    """
    if custom_names is not None:
        if isinstance(custom_names, Mapping) and agent_id in custom_names:
            return custom_names[agent_id]
        if index is not None and not isinstance(custom_names, Mapping):
            return custom_names[index]

    if agent_id.startswith("agent_"):
        suffix = agent_id[len("agent_") :]
        if suffix.isdigit():
            return f"Agent {suffix}"

    return agent_id
