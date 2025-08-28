from aiops.config import Config
from aiops.orchestrator.pipeline import AirportOpsPipeline


def test_pipeline_runs():
	cfg = Config(num_flights=12, num_runways=2, num_gates=6)
	p = AirportOpsPipeline(cfg)
	out = p.run_once()
	assert out.schedule.status in {"Optimal", "Feasible", "Not Solved"}
	assert len(out.schedule.runway_assignments) == cfg.num_flights
	assert len(out.schedule.gate_assignments) == cfg.num_flights

