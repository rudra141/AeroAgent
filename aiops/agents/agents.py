from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from aiops.agents.core import Agent, Event, EventBus


@dataclass
class WorldState:
	flights: pd.DataFrame
	runways: List[str]
	gates: pd.DataFrame
	weather: pd.DataFrame
	runway_assignments: Dict[str, int]  # flight_id -> slot
	gate_assignments: Dict[str, str]    # flight_id -> gate_id
	delays: Dict[str, float]


class WeatherAgent(Agent):
	def __init__(self, bus: EventBus, weather: pd.DataFrame):
		super().__init__("WeatherAgent", bus)
		self.weather = weather

	def step(self, t_min: int) -> None:
		row_idx = min(len(self.weather) - 1, t_min // 5)
		wx = self.weather.iloc[row_idx]
		self.bus.publish(Event("weather.update", {"wind": float(wx["wind"]), "rain": int(wx["rain"])}, t_min))


class FlightAgent(Agent):
	def __init__(self, bus: EventBus, flights: pd.DataFrame, delays: Dict[str, float], bias: Dict[str, float] | None = None):
		super().__init__("FlightAgent", bus)
		self.flights = flights
		self.delays = delays
		self.bias: Dict[str, float] = {} if bias is None else dict(bias)
		self.wx_extra_delay_min: float = 0.0
		bus.subscribe("runway.deferred", self.on_deferred)
		bus.subscribe("weather.update", self.on_weather)

	def step(self, t_min: int) -> None:
		# Emit requests for flights scheduled within next 10 minutes
		window = self.flights[(self.flights["sched_min"] >= t_min) & (self.flights["sched_min"] < t_min + 10)]
		for _, row in window.iterrows():
			base = int(row["sched_min"]) + int(self.delays.get(row["flight_id"], 0))
			bias = int(self.bias.get(row["flight_id"], 0))
			req_sched = max(0, base + bias + int(self.wx_extra_delay_min))
			req = {"flight_id": row["flight_id"], "op": row["op"], "sched_min": req_sched}
			self.bus.publish(Event("flight.request", req, t_min))
			self.record_decision()

	def on_deferred(self, evt: Event) -> None:
		fid = evt.payload["flight_id"]
		delta_slots = int(evt.payload.get("delta_slots", 0))
		# Learn by increasing bias when deferrals occur (half the deferral time in minutes)
		self.bias[fid] = self.bias.get(fid, 0.0) + delta_slots * 5 * 0.5

	def on_weather(self, evt: Event) -> None:
		# Simple adaptive rule: when raining, expect +2 minutes extra
		rain = int(evt.payload.get("rain", 0))
		self.wx_extra_delay_min = 2.0 if rain == 1 else 0.0


class RunwayAgent(Agent):
	def __init__(self, bus: EventBus, runways: List[str]) -> None:
		super().__init__("RunwayAgent", bus)
		self.runways = runways
		self.occupied: Dict[int, str] = {}
		bus.subscribe("flight.request", self.on_request)

	def on_request(self, evt: Event) -> None:
		# Greedy assign earliest free runway at requested or next slot
		req_slot = evt.payload["sched_min"] // 5
		for s in [req_slot, req_slot + 1, req_slot + 2]:
			for r in self.runways:
				if (s, r) not in self.occupied:
					self.occupied[(s, r)] = evt.payload["flight_id"]
					self.bus.publish(Event("runway.assigned", {"flight_id": evt.payload["flight_id"], "runway_id": r, "slot": s}, evt.time_min))
					if s > req_slot:
						self.bus.publish(Event("runway.deferred", {"flight_id": evt.payload["flight_id"], "delta_slots": s - req_slot}, evt.time_min))
					self.record_decision()
					return


class GateAgent(Agent):
	def __init__(self, bus: EventBus, gates: pd.DataFrame) -> None:
		super().__init__("GateAgent", bus)
		self.gates = list(gates["gate_id"]) if not gates.empty else ["G-1"]
		self.occupied: Dict[int, str] = {}
		bus.subscribe("runway.assigned", self.on_runway_assigned)

	def on_runway_assigned(self, evt: Event) -> None:
		flight_id = evt.payload["flight_id"]
		slot = int(evt.payload["slot"])
		for s in [slot, slot + 1, slot + 2]:
			for g in self.gates:
				if (s, g) not in self.occupied:
					self.occupied[(s, g)] = flight_id
					self.bus.publish(Event("gate.assigned", {"flight_id": flight_id, "gate_id": g, "slot": s}, evt.time_min))
					self.record_decision()
					return


