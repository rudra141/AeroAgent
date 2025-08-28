import json
import click

from aiops.config import Config
from aiops.orchestrator.pipeline import AirportOpsPipeline
from aiops.orchestrator.sim import Simulation


@click.group()
def main() -> None:
	"""AI Ops Workflow CLI."""
	pass


@main.command()
@click.option("--flights", type=int, default=40, show_default=True)
@click.option("--runways", type=int, default=2, show_default=True)
@click.option("--gates", type=int, default=10, show_default=True)
def run(flights: int, runways: int, gates: int) -> None:
	"""Run one end-to-end optimization cycle."""
	cfg = Config(num_flights=flights, num_runways=runways, num_gates=gates)
	pipeline = AirportOpsPipeline(cfg)
	output = pipeline.run_once()
	result = {
		"status": output.schedule.status,
		"objective": output.schedule.objective_value,
		"mean_predicted_delay": output.mean_predicted_delay,
		"latency_seconds": output.latency_seconds,
		"num_runway_assignments": int(len(output.schedule.runway_assignments)),
		"num_gate_assignments": int(len(output.schedule.gate_assignments)),
		"num_taxiway_conflicts": int(len(output.schedule.taxiway_conflicts)),
		"alerts": [
			{"type": a.type, "message": a.message, "meta": a.meta}
			for a in output.alerts
		],
	}
	click.echo(json.dumps(result, indent=2))


@main.command()
@click.option("--minutes", type=int, default=120, show_default=True)
def simulate(minutes: int) -> None:
	"""Run a simple multi-agent simulation and print metrics."""
	cfg = Config()
	sim = Simulation(cfg)
	metrics = sim.run(horizon_minutes=minutes)
	res = {
		"decisions": metrics.decisions,
		"events": metrics.num_events,
		"assignments": len(metrics.assignments),
	}
	click.echo(json.dumps(res, indent=2))


@main.command()
@click.option("--episodes", type=int, default=3, show_default=True)
@click.option("--minutes", type=int, default=120, show_default=True)
def simulate_learn(episodes: int, minutes: int) -> None:
	"""Run multiple simulations with adaptive learning and print per-episode stats."""
	cfg = Config()
	sim = Simulation(cfg)
	result = sim.run_with_learning(episodes=episodes, horizon_minutes=minutes)
	click.echo(json.dumps(result, indent=2))


