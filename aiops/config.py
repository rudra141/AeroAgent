from dataclasses import dataclass


@dataclass
class Config:
	# Simulation sizes
	num_flights: int = 40
	num_runways: int = 2
	num_gates: int = 10
	seed: int = 42

	# Time resolution in minutes for the schedule grid
	time_slot_minutes: int = 5
	planning_horizon_minutes: int = 240

	# Simple delay model parameters
	base_delay_mean: float = 3.0
	base_delay_std: float = 4.0
	weather_delay_multiplier: float = 1.5

	# Alerting threshold (minutes)
	delay_alert_threshold: int = 15

	# Operational constraints
	runway_separation_minutes: int = 5  # minimum time gap on same runway
	gate_turnaround_minutes: int = 15    # minimum gap before reusing same gate
	enable_runway_separation: bool = False
	enable_gate_turnaround: bool = False

