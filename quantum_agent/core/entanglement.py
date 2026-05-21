"""Agent entanglement — shared state propagation between agents.

When two agents are *entangled*, a state change in one instantly updates
a shared bus that the other can observe.  This enables cooperative
reasoning without explicit message-passing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EntanglementEvent:
    source_agent: str
    key: str
    value: Any
    timestamp: float = 0.0


Listener = Callable[[EntanglementEvent], None]


class EntanglementBus:
    """Pub/sub bus for entangled agent state.

    Agents subscribe to keys they care about.  When any agent publishes a
    value, all subscribers (except the publisher) receive the event
    synchronously.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._listeners: dict[str, list[tuple[str, Listener]]] = {}
        self._event_log: list[EntanglementEvent] = []
        self._counter: int = 0

    def publish(self, source_agent: str, key: str, value: Any) -> None:
        self._counter += 1
        event = EntanglementEvent(
            source_agent=source_agent,
            key=key,
            value=value,
            timestamp=float(self._counter),
        )
        self._state[key] = value
        self._event_log.append(event)

        for agent_id, callback in self._listeners.get(key, []):
            if agent_id != source_agent:
                callback(event)

    def subscribe(self, agent_id: str, key: str, callback: Listener) -> None:
        self._listeners.setdefault(key, []).append((agent_id, callback))

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    @property
    def event_log(self) -> list[EntanglementEvent]:
        return list(self._event_log)

    def __repr__(self) -> str:
        return f"EntanglementBus(keys={list(self._state.keys())})"
