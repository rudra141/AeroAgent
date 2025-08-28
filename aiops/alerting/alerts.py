from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from aiops.config import Config


@dataclass
class Alert:
	type: str
	message: str
	meta: Dict[str, str]


class AlertAgent:
	def __init__(self, config: Config) -> None:
		self.config = config

	def generate(self, delays: pd.Series, runway_assign: pd.DataFrame) -> List[Alert]:
		alerts: List[Alert] = []
		# Delay threshold alerts
		for flight_id, d in delays.items():
			if float(d) >= self.config.delay_alert_threshold:
				alerts.append(Alert(
					type="DELAY_THRESHOLD",
					message=f"Flight {flight_id} predicted delay {float(d):.1f} min exceeds threshold",
					meta={"flight_id": flight_id, "predicted_delay_min": f"{float(d):.1f}"},
				))

		# Simple runway conflict alerts: same runway and adjacent slots (tight spacing)
		if not runway_assign.empty:
			df = runway_assign.copy().sort_values(["runway_id", "slot"]).reset_index(drop=True)
			for i in range(1, len(df)):
				prev = df.iloc[i - 1]
				cur = df.iloc[i]
				if prev["runway_id"] == cur["runway_id"] and (cur["slot"] - prev["slot"]) <= 0:
					alerts.append(Alert(
						type="RUNWAY_CONFLICT",
						message=f"Potential runway conflict: {prev['flight_id']} and {cur['flight_id']} on {cur['runway_id']} at slot {cur['slot']}",
						meta={"runway": str(cur["runway_id"]), "slot": str(int(cur["slot"]))},
					))

		return alerts


