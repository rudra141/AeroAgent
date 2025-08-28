# AI Workflow for Airport Operations (Mini Project)

## Quickstart

```bash
# From this folder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Or: pip install -e .

# Run a single workflow cycle
python -m aiops.cli run --flights 30 --runways 2 --gates 10
```

### Dashboard (Streamlit)

```bash
source .venv/bin/activate
streamlit run aiops/ui/app.py
```

## What it does

- Simulates flights, runways, gates, and weather
- Trains a small delay prediction model (RandomForest)
- Optimizes runway and gate assignments via MILP (PuLP)
- Emits JSON summary with objective, status, and counts

## Project structure

- `aiops/ingestion`: synthetic data generator
- `aiops/prediction`: delay model
- `aiops/optimization`: runway+gate scheduler
- `aiops/orchestrator`: pipeline that ties it all together
- `aiops/cli.py`: CLI entrypoint
- `tests/`: smoke test

## Notes

- This is a teaching/demo scaffold, not production-ready
- Replace synthetic data with real feeds and extend constraints as needed


