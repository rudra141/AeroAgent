from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple

from aiops.config import Config


@dataclass
class DelayPredictions:
	per_flight_minutes: pd.Series
	mean_delay: float


class DelayPredictionModel:
	def __init__(self, config: Config) -> None:
		self.config = config
		self.model = RandomForestRegressor(n_estimators=100, random_state=config.seed)
		self._is_trained = False

	def _make_features(self, flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
		# Simple join on nearest time slot
		weather_idx = (flights["sched_min"] // self.config.time_slot_minutes).clip(0, len(weather) - 1)
		wx = weather.iloc[weather_idx.values].reset_index(drop=True)
		X = pd.DataFrame(
			{
				"sched_min": flights["sched_min"].values,
				"is_arr": (flights["op"] == "ARR").astype(int).values,
				"is_wide": (flights["aircraft"] == "WIDE").astype(int).values,
				"wind": wx["wind"].values,
				"rain": wx["rain"].values,
			}
		)
		return X

	def fit(self, flights: pd.DataFrame, weather: pd.DataFrame) -> None:
		np.random.seed(self.config.seed)
		X = self._make_features(flights, weather)
		# Synthetic target: base + rain penalty + wide-body handling variance
		y = (
			np.random.normal(self.config.base_delay_mean, self.config.base_delay_std, size=len(X))
			+ X["rain"].values * self.config.weather_delay_multiplier * 2.0
			+ X["is_wide"].values * 1.5
		)
		self.model.fit(X, y)
		self._is_trained = True

	def predict(self, flights: pd.DataFrame, weather: pd.DataFrame) -> DelayPredictions:
		if not self._is_trained:
			raise RuntimeError("Model must be fit before predict().")
		X = self._make_features(flights, weather)
		pred = self.model.predict(X)
		return DelayPredictions(per_flight_minutes=pd.Series(pred, index=flights["flight_id"]), mean_delay=float(np.mean(pred)))


