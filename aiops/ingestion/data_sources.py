from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from aiops.config import Config


@dataclass
class IngestedData:
	flights: pd.DataFrame
	runways: pd.DataFrame
	gates: pd.DataFrame
	weather: pd.DataFrame


class DataIngestion:
	def __init__(self, config: Config) -> None:
		self.config = config

	def simulate(self) -> IngestedData:
		np.random.seed(self.config.seed)

		flight_ids = [f"FL{1000 + i}" for i in range(self.config.num_flights)]
		scheduled_times = np.sort(
			np.random.randint(
				0,
				self.config.planning_horizon_minutes,
				size=self.config.num_flights,
			),
		)
		arr_or_dep = np.random.choice(["ARR", "DEP"], size=self.config.num_flights, p=[0.5, 0.5])
		aircraft_type = np.random.choice(["NARROW", "WIDE"], size=self.config.num_flights, p=[0.8, 0.2])

		flights = pd.DataFrame(
			{
				"flight_id": flight_ids,
				"sched_min": scheduled_times,
				"op": arr_or_dep,
				"aircraft": aircraft_type,
			}
		)

		runways = pd.DataFrame({"runway_id": [f"RWY-{i+1}" for i in range(self.config.num_runways)]})
		gates = pd.DataFrame({"gate_id": [f"G-{i+1}" for i in range(self.config.num_gates)], "compatible": np.random.choice(["NARROW", "WIDE"], size=self.config.num_gates, p=[0.7, 0.3])})

		weather_time = np.arange(0, self.config.planning_horizon_minutes, self.config.time_slot_minutes)
		wind = np.random.normal(10, 3, size=len(weather_time))
		rain = np.random.binomial(1, 0.2, size=len(weather_time))
		weather = pd.DataFrame({"minute": weather_time, "wind": wind, "rain": rain})

		return IngestedData(flights=flights, runways=runways, gates=gates, weather=weather)


