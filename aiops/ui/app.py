from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import zipfile

from aiops.config import Config
from aiops.orchestrator.pipeline import AirportOpsPipeline
from aiops.orchestrator.sim import Simulation


st.set_page_config(page_title="Airport Ops AI Workflow", layout="wide")
st.title("Airport Operations AI Workflow")

with st.sidebar:
	st.header("Parameters")
	num_flights = st.slider("Flights", 10, 100, 30, step=2)
	num_runways = st.slider("Runways", 1, 4, 2)
	num_gates = st.slider("Gates", 2, 20, 10)
	seed = st.number_input("Seed", min_value=0, value=42, step=1)
	st.divider()
	st.header("Constraints")
	enable_runway_sep = st.checkbox("Enforce runway separation", value=False)
	runway_sep = st.slider("Runway separation (min)", 1, 10, 5)
	enable_gate_turn = st.checkbox("Enforce gate turnaround", value=False)
	gate_turn = st.slider("Gate turnaround (min)", 5, 60, 15, step=5)
	if st.button("Run Workflow"):
		cfg = Config(
			num_flights=num_flights,
			num_runways=num_runways,
			num_gates=num_gates,
			seed=seed,
			enable_runway_separation=enable_runway_sep,
			runway_separation_minutes=runway_sep,
			enable_gate_turnaround=enable_gate_turn,
			gate_turnaround_minutes=gate_turn,
		)
		pipeline = AirportOpsPipeline(cfg)
		out = pipeline.run_once()
		st.session_state["output"] = out

out = st.session_state.get("output")
if out is not None:
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Status", out.schedule.status)
		st.metric("Objective", f"{out.schedule.objective_value:.2f}")
	with col2:
		st.metric("Mean Pred Delay (min)", f"{out.mean_predicted_delay:.2f}")
		st.metric("Latency (s)", f"{out.latency_seconds:.2f}")
	with col3:
		st.metric("Flights", len(out.schedule.runway_assignments))
		st.metric("Taxiway Conflicts", len(out.schedule.taxiway_conflicts))

	st.subheader("Runway Assignments")
	st.dataframe(out.schedule.runway_assignments)

	# Timeline chart for runway schedule
	if not out.schedule.runway_assignments.empty:
		df_tl = out.schedule.runway_assignments.copy()
		# Convert 5-minute slots to minutes since t0, then to datetimes for nicer axis labels
		df_tl["start_min"] = df_tl["slot"] * 5
		df_tl["end_min"] = df_tl["start_min"] + 5
		anchor = pd.Timestamp("2000-01-01 00:00:00")
		df_tl["start"] = pd.to_datetime(df_tl["start_min"], unit="m", origin=anchor)
		df_tl["end"] = pd.to_datetime(df_tl["end_min"], unit="m", origin=anchor)
		fig = px.timeline(
			df_tl,
			x_start="start",
			x_end="end",
			y="runway_id",
			color="flight_id",
			title="Runway Schedule (5-min slots)",
		)
		fig.update_yaxes(autorange="reversed")
		fig.update_layout(xaxis_title="Time")
		st.plotly_chart(fig, use_container_width=True)

	st.subheader("Gate Assignments")
	st.dataframe(out.schedule.gate_assignments)

	if not out.schedule.taxiway_conflicts.empty:
		st.subheader("Taxiway Conflicts")
		st.warning("Potential conflicts detected")
		df_conf = out.schedule.taxiway_conflicts.copy()
		st.dataframe(df_conf)

	if out.alerts:
		st.subheader("Alerts")
		for a in out.alerts:
			st.error(f"[{a.type}] {a.message}")

	# Highlight conflicting rows in runway assignment table
	if not out.schedule.taxiway_conflicts.empty and not out.schedule.runway_assignments.empty:
		conf_pairs = set()
		for _, row in out.schedule.taxiway_conflicts.iterrows():
			conf_pairs.add((row["flight_id_a"], row["runway_id"], row["slot"]))
			conf_pairs.add((row["flight_id_b"], row["runway_id"], row["slot"]))
		df = out.schedule.runway_assignments.copy()
		def row_style(r):
			key = (r["flight_id"], r["runway_id"], r["slot"])
			return ["background-color: #ffe6e6" if key in conf_pairs else ""] * len(r)
		st.subheader("Runway Assignments (highlight conflicts)")
		st.dataframe(df.style.apply(row_style, axis=1))

else:
	st.info("Set parameters and click 'Run Workflow' in the sidebar.")

st.divider()
st.header("Multi-Agent Simulation")
sim_minutes = st.slider("Simulation horizon (minutes)", 30, 480, 120, step=30)
if st.button("Run Simulation"):
	sim = Simulation(Config())
	metrics = sim.run(horizon_minutes=sim_minutes)
	st.subheader("Agent Decisions")
	st.json(metrics.decisions)
	st.metric("Events", metrics.num_events)
	if not metrics.assignments.empty:
		st.subheader("Assignments from Simulation")
		st.dataframe(metrics.assignments)
		# Build a simple timeline figure from simulation assignments
		df_sim = metrics.assignments.copy()
		df_sim["start_min"] = df_sim["slot"] * 5
		df_sim["end_min"] = df_sim["start_min"] + 5
		anchor = pd.Timestamp("2000-01-01 00:00:00")
		df_sim["start"] = pd.to_datetime(df_sim["start_min"], unit="m", origin=anchor)
		df_sim["end"] = pd.to_datetime(df_sim["end_min"], unit="m", origin=anchor)
		fig_sim = px.timeline(df_sim, x_start="start", x_end="end", y="runway_id", color="flight_id", title="Simulation Runway Schedule")
		fig_sim.update_yaxes(autorange="reversed")
		st.plotly_chart(fig_sim, use_container_width=True)
	logs = getattr(metrics, "logs", [])
	if logs:
		st.subheader("Agent Event Log")
		log_df = pd.DataFrame(logs)
		st.dataframe(log_df.tail(100))

		# Export report bundle (ZIP)
		buf = io.BytesIO()
		with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
			z.writestr("logs.csv", log_df.to_csv(index=False))
			z.writestr("assignments.csv", metrics.assignments.to_csv(index=False))
			z.writestr("timeline.html", pio.to_html(fig_sim, full_html=True, include_plotlyjs="cdn"))
		st.download_button("Download simulation report (ZIP)", data=buf.getvalue(), file_name="simulation_report.zip", mime="application/zip")

st.subheader("Adaptive Learning (multi-episode)")
episodes = st.slider("Episodes", 1, 10, 3)
if st.button("Run Learning Simulation"):
	sim = Simulation(Config())
	res = sim.run_with_learning(episodes=episodes, horizon_minutes=sim_minutes)
	st.json(res)
	# Plot improvement proxy: events and assignments per episode
	hist = pd.DataFrame(res["episodes"])
	if not hist.empty:
		c1, c2 = st.columns(2)
		with c1:
			st.subheader("Events per Episode")
			st.bar_chart(hist.set_index("episode")["events"])
		with c2:
			st.subheader("Assignments per Episode")
			st.bar_chart(hist.set_index("episode")["assignments"])

		# Export learning report bundle (ZIP)
		fig_events = px.bar(hist, x="episode", y="events", title="Events per Episode")
		fig_assign = px.bar(hist, x="episode", y="assignments", title="Assignments per Episode")
		buf2 = io.BytesIO()
		with zipfile.ZipFile(buf2, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
			z.writestr("episodes.csv", hist.to_csv(index=False))
			final_bias = pd.DataFrame(list(res.get("final_bias", {}).items()), columns=["flight_id", "bias_min"]) if res.get("final_bias") else pd.DataFrame(columns=["flight_id", "bias_min"])
			z.writestr("final_bias.csv", final_bias.to_csv(index=False))
			z.writestr("events_bar.html", pio.to_html(fig_events, full_html=True, include_plotlyjs="cdn"))
			z.writestr("assignments_bar.html", pio.to_html(fig_assign, full_html=True, include_plotlyjs="cdn"))
		st.download_button("Download learning report (ZIP)", data=buf2.getvalue(), file_name="learning_report.zip", mime="application/zip")


