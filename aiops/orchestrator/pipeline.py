from __future__ import annotations

import time
from dataclasses import dataclass

from aiops.config import Config
from aiops.utils.logging import get_logger
from aiops.ingestion.data_sources import DataIngestion
from aiops.prediction.models import DelayPredictionModel
from aiops.optimization.schedulers import RunwayGateScheduler, ScheduleResult
from aiops.alerting.alerts import AlertAgent, Alert


logger = get_logger(__name__)


@dataclass
class PipelineOutput:
	schedule: ScheduleResult
	mean_predicted_delay: float
	latency_seconds: float
	alerts: list[Alert]


class AirportOpsPipeline:
	def __init__(self, config: Config) -> None:
		self.config = config
		self.ingestion = DataIngestion(config)
		self.predictor = DelayPredictionModel(config)
		self.scheduler = RunwayGateScheduler(
			time_slot_minutes=config.time_slot_minutes,
			runway_sep_minutes=config.runway_separation_minutes,
			gate_turnaround_minutes=config.gate_turnaround_minutes,
			enable_runway_sep=config.enable_runway_separation,
			enable_gate_turn=config.enable_gate_turnaround,
		)
		self.alerter = AlertAgent(config)

	def run_once(self) -> PipelineOutput:
		start = time.time()
		logger.info("Ingesting data...")
		data = self.ingestion.simulate()

		logger.info("Training prediction model...")
		self.predictor.fit(data.flights, data.weather)
		pred = self.predictor.predict(data.flights, data.weather)
		logger.info("Predicted mean delay: %.2f min", pred.mean_delay)

		logger.info("Optimizing runway and gate assignments...")
		schedule = self.scheduler.optimize(data.flights, data.runways, data.gates, pred.per_flight_minutes)
		alerts = self.alerter.generate(pred.per_flight_minutes, schedule.runway_assignments)
		latency = time.time() - start
		logger.info("Optimization status: %s | objective=%.2f | latency=%.2fs | alerts=%d", schedule.status, schedule.objective_value, latency, len(alerts))
		return PipelineOutput(schedule=schedule, mean_predicted_delay=pred.mean_delay, latency_seconds=latency, alerts=alerts)


