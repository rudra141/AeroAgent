from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Callable, Any


@dataclass
class Event:
	type: str
	payload: Dict[str, Any]
	time_min: int


class EventBus:
	def __init__(self) -> None:
		self.subscribers: Dict[str, List[Callable[[Event], None]]] = {}
		self.log: List[Event] = []

	def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
		self.subscribers.setdefault(event_type, []).append(handler)

	def publish(self, evt: Event) -> None:
		self.log.append(evt)
		for handler in self.subscribers.get(evt.type, []):
			handler(evt)


class Agent:
	def __init__(self, name: str, bus: EventBus) -> None:
		self.name = name
		self.bus = bus
		self.decisions = 0

	def step(self, t_min: int) -> None:
		pass

	def record_decision(self) -> None:
		self.decisions += 1


