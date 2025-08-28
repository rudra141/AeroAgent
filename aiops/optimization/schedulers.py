from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp


@dataclass
class ScheduleResult:
	runway_assignments: pd.DataFrame  # columns: flight_id, runway_id, slot
	gate_assignments: pd.DataFrame    # columns: flight_id, gate_id
	taxiway_conflicts: pd.DataFrame   # columns: flight_id_a, flight_id_b, runway_id, slot
	objective_value: float
	status: str



class RunwayGateScheduler:
	def __init__(self, time_slot_minutes: int, runway_sep_minutes: int = 5, gate_turnaround_minutes: int = 15, enable_runway_sep: bool = True, enable_gate_turn: bool = True) -> None:
		self.slot_minutes = time_slot_minutes
		self.runway_sep_slots = max(1, runway_sep_minutes // time_slot_minutes)
		self.gate_turn_slots = max(1, gate_turnaround_minutes // time_slot_minutes)
		self.enable_runway_sep = enable_runway_sep
		self.enable_gate_turn = enable_gate_turn

	def _time_to_slot(self, minute: int) -> int:
		return int(minute // self.slot_minutes)

	def optimize(self, flights: pd.DataFrame, runways: pd.DataFrame, gates: pd.DataFrame, delay_minutes: pd.Series) -> ScheduleResult:
		# Discretize time into slots and create a small MILP
		flights = flights.copy().reset_index(drop=True)
		flights["slot"] = flights["sched_min"].apply(self._time_to_slot)
		# Candidate slots per flight: allow s-1, s, s+1 within bounds
		min_slot = max(0, int(flights["slot"].min()) - 1)
		max_slot = int(flights["slot"].max()) + 1

		rwy_list = list(runways["runway_id"]) if len(runways) > 0 else ["RWY-1"]
		gate_list = list(gates["gate_id"]) if len(gates) > 0 else ["G-1"]
		compat = {row["gate_id"]: row["compatible"] for _, row in gates.iterrows()}

		prob = pulp.LpProblem("RunwayGateScheduling", pulp.LpMinimize)

		# Variables
		x = pulp.LpVariable.dicts(
			"x",
			((i, r, s)
			 for i in flights.index
			 for r in rwy_list
			 for s in [s for s in [flights.loc[i, "slot"] - 1, flights.loc[i, "slot"], flights.loc[i, "slot"] + 1] if s >= min_slot and s <= max_slot]),
			0, 1, pulp.LpBinary,
		)
		g = pulp.LpVariable.dicts("g", ((i, k) for i in flights.index for k in gate_list), 0, 1, pulp.LpBinary)

		# Objective: minimize delay proximity + simple spreading on runways
		flight_delay = delay_minutes.reindex(flights["flight_id"]).fillna(delay_minutes.mean())
		objective_terms = []
		for i in flights.index:
			s_sched = flights.loc[i, "slot"]
			for r in rwy_list:
				for s in [s for s in [s_sched - 1, s_sched, s_sched + 1] if s >= min_slot and s <= max_slot]:
					# penalty: moving from scheduled slot + predicted delay
					objective_terms.append((abs(s - s_sched) + float(flight_delay.iloc[i]) / 10.0) * x[(i, r, s)])
		prob += pulp.lpSum(objective_terms)

		# Constraints
		# Each flight assigned to exactly one runway slot
		for i in flights.index:
			cand = [s for s in [flights.loc[i, "slot"] - 1, flights.loc[i, "slot"], flights.loc[i, "slot"] + 1] if s >= min_slot and s <= max_slot]
			prob += pulp.lpSum(x[(i, r, s)] for r in rwy_list for s in cand) == 1

		# Runway capacity + separation: at most 1 per slot and enforce separation window
		all_slots = list(range(min_slot, max_slot + 1))
		for r in rwy_list:
			for s in all_slots:
				# capacity
				prob += pulp.lpSum(x[(i, r, s)] for i in flights.index if (i, r, s) in x) <= 1
				# separation: if enabled, block adjacent slots within sep window
				if self.enable_runway_sep:
					for d in range(1, self.runway_sep_slots + 1):
						if (s + d) in all_slots:
							prob += (
								pulp.lpSum(x[(i, r, s)] for i in flights.index if (i, r, s) in x)
								+
								pulp.lpSum(x[(i, r, s + d)] for i in flights.index if (i, r, s + d) in x)
								<= 1
							)

		# Gate assignment: one gate per flight, must match compatibility
		for i in flights.index:
			aircraft = flights.loc[i, "aircraft"]
			compatible_gates = [k for k in gate_list if compat.get(k, "NARROW") == aircraft or (compat.get(k, "NARROW") == "WIDE" and aircraft == "NARROW")]
			if not compatible_gates:
				# if no compatible gates, allow any gate (fallback)
				compatible_gates = gate_list
			prob += pulp.lpSum(g[(i, k)] for k in compatible_gates) == 1
			# disallow incompatible explicitly
			for k in gate_list:
				if k not in compatible_gates:
					prob += g[(i, k)] == 0

		# Gate capacity with turnaround: enforce gap between flights on same gate
		for k in gate_list:
			for s in all_slots:
				# capacity at s (one per bucket)
				prob += pulp.lpSum(g[(i, k)] for i in flights.index if flights.loc[i, "slot"] == s) <= 1
				# turnaround window: if enabled, block s+d buckets
				if self.enable_gate_turn:
					for d in range(1, self.gate_turn_slots + 1):
						if (s + d) in all_slots:
							prob += (
								pulp.lpSum(g[(i, k)] for i in flights.index if flights.loc[i, "slot"] == s)
								+
								pulp.lpSum(g[(i, k)] for i in flights.index if flights.loc[i, "slot"] == s + d)
								<= 1
							)

		status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
		status_str = pulp.LpStatus[prob.status]

		# Extract solutions (only if solved feasibility)
		runway_rows = []
		gate_rows = []
		if status_str in {"Optimal", "Feasible"}:
			for i in flights.index:
				for r in rwy_list:
					for s in all_slots:
						if (i, r, s) in x and pulp.value(x[(i, r, s)]) > 0.5:
							runway_rows.append({"flight_id": flights.loc[i, "flight_id"], "runway_id": r, "slot": int(s)})
			for i in flights.index:
				for k in gate_list:
					if pulp.value(g[(i, k)]) > 0.5:
						gate_rows.append({"flight_id": flights.loc[i, "flight_id"], "gate_id": k})


		# Basic taxiway conflict heuristic: same runway same slot implies potential conflict pair
		runway_df = pd.DataFrame(runway_rows)
		taxi_conf_rows = []
		if not runway_df.empty:
			grp = runway_df.groupby(["runway_id", "slot"]).agg(list)
			for (r, s), row in grp.iterrows():
				fls = row["flight_id"]
				if isinstance(fls, list) and len(fls) > 1:
					for i in range(len(fls)):
						for j in range(i + 1, len(fls)):
							taxi_conf_rows.append({"flight_id_a": fls[i], "flight_id_b": fls[j], "runway_id": r, "slot": int(s)})

		return ScheduleResult(
			runway_assignments=runway_df,
			gate_assignments=pd.DataFrame(gate_rows),
			taxiway_conflicts=pd.DataFrame(taxi_conf_rows),
			objective_value=pulp.value(prob.objective) if prob.status == 1 else float("inf"),
			status=status_str,
		)


