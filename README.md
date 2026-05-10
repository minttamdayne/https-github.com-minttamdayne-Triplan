# Triplan

Triplan is an AI trip planner for Ho Chi Minh City. It generates multi-day itineraries from travel preferences, enriches date-specific festivals, routes daily stops, and includes a small browser frontend served by FastAPI.

## Requirements

- Python 3.11+
- Ollama for local LLM usage
- Optional: Neo4j for the knowledge graph
- Optional: Google Maps API key for geocoding
- Optional: Tavily API key for web search

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` with your own keys. Do not commit real API keys or passwords.

For local LLM usage, install Ollama and pull the default model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running:

```bash
ollama serve
```

## Run The App

Start the FastAPI server:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the frontend:

```text
http://127.0.0.1:8000
```

API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Try Festival Results

Festival candidates only appear when the trip dates overlap events in `data/HCM_FEST.json`.

Good test dates:

- `2026-03-26` to `2026-03-29`
- `2026-04-15` to `2026-04-18`
- `2026-11-11` to `2026-11-14`

If you choose dates with no matching festival data, the itinerary will contain regular POIs only.

## Run Tests

Run the focused tests:

```bash
pytest tests/test_routing_festival.py tests/test_opening_hours.py tests/test_natural_language.py
```

Run all tests:

```bash
pytest
```

## Useful Scripts

Run the full agent pipeline from the terminal:

```bash
python scripts/test_agents.py --start-date 2026-03-26 --days 4 --debug
```

Seed the knowledge graph:

```bash
python scripts/seed_knowledge_graph.py
```

## Project Layout

```text
frontend/              Browser UI served by FastAPI
src/api/               FastAPI app and routes
src/agents/            Planning, scoring, festival, routing, and chat agents
src/models/            Pydantic request and response models
src/tools/             Geocoding, web search, distance, and KG helpers
data/                  POI and festival data
tests/                 Pytest coverage
scripts/               Local debug and data scripts
```

## Notes

- Keep real secrets in `.env`, not `.env.example`.
- The frontend posts to `/api/v1/plan` and `/api/v1/chat`.
- `python main.py` starts Uvicorn with reload. If reload causes watcher issues, use the `python -m uvicorn ...` command above.
