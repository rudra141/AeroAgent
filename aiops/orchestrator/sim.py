from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd

from aiops.config import Config
from aiops.ingestion.data_sources import DataIngestion
from aiops.prediction.models import DelayPredictionModel
from aiops.agents.core import EventBus
from aiops.agents.agents import WeatherAgent, FlightAgent, RunwayAgent, GateAgent


@dataclass
class SimulationMetrics:
	decisions: Dict[str, int]
	num_events: int
	assignments: pd.DataFrame
	logs: List[Dict[str, Any]]


class Simulation:
	def __init__(self, config: Config) -> None:
		self.config = config
		self.bus = EventBus()
		self.metrics = SimulationMetrics(decisions={}, num_events=0, assignments=pd.DataFrame(), logs=[])

	def run(self, horizon_minutes: int = 240, step_minutes: int = 5) -> SimulationMetrics:
		# Prepare data and delays
		data = DataIngestion(self.config).simulate()
		pred = DelayPredictionModel(self.config)
		pred.fit(data.flights, data.weather)
		delay_series = pred.predict(data.flights, data.weather).per_flight_minutes.to_dict()

		# Agents
		weather = WeatherAgent(self.bus, data.weather)
		flights = FlightAgent(self.bus, data.flights, delay_series)
		runways = RunwayAgent(self.bus, list(data.runways["runway_id"]))
		gates = GateAgent(self.bus, data.gates)

		# Simulate
		for t in range(0, horizon_minutes, step_minutes):
			weather.step(t)
			flights.step(t)
			# Runway and Gate agents react via events

		# Aggregate results
		rows = []
		for (slot, rwy), fid in runways.occupied.items():
			rows.append({"flight_id": fid, "runway_id": rwy, "slot": slot})
		assign_df = pd.DataFrame(rows)
		self.metrics.assignments = assign_df
		self.metrics.decisions = {
			"WeatherAgent": weather.decisions,
			"FlightAgent": flights.decisions,
			"RunwayAgent": runways.decisions,
			"GateAgent": gates.decisions,
		}
		self.metrics.num_events = len(self.bus.log)
		self.metrics.logs = [{"time_min": e.time_min, "type": e.type, **{k: v for k, v in e.payload.items()}} for e in self.bus.log]
		return self.metrics

	def run_with_learning(self, episodes: int = 3, horizon_minutes: int = 240, step_minutes: int = 5) -> Dict[str, any]:
		history = []
		bias: Dict[str, float] = {}
		for ep in range(episodes):
			data = DataIngestion(self.config).simulate()
			pred = DelayPredictionModel(self.config)
			pred.fit(data.flights, data.weather)
			delay_series = pred.predict(data.flights, data.weather).per_flight_minutes.to_dict()

			bus = EventBus()
			weather = WeatherAgent(bus, data.weather)
			flights = FlightAgent(bus, data.flights, delay_series, bias=bias)
			runways = RunwayAgent(bus, list(data.runways["runway_id"]))
			gates = GateAgent(bus, data.gates)

			for t in range(0, horizon_minutes, step_minutes):
				weather.step(t)
				flights.step(t)

			# update bias from learned deferrals
			bias.update(flights.bias)

			history.append({
				"episode": ep + 1,
				"decisions": {
					"WeatherAgent": weather.decisions,
					"FlightAgent": flights.decisions,
					"RunwayAgent": runways.decisions,
					"GateAgent": gates.decisions,
				},
				"events": len(bus.log),
				"assignments": len(runways.occupied),
				"avg_bias_min": float(pd.Series(list(bias.values())).mean()) if bias else 0.0,
			})

		return {"episodes": history, "final_bias": bias}


