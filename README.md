# URITOMO Backend

Backend for real-time multilingual meetings, LiveKit-based sessions, and translation/summarization workflows.

## Overview

URITOMO backend provides:
- User authentication and profile data for meeting apps
- Room/meeting lifecycle and live session tracking
- WebSocket channel for real-time session chat
- Translation API (DeepL or mock) with event storage
- Summary endpoints (currently mock responses)
- LiveKit token issuance for clients and workers
- Optional background workers (RQ + LiveKit realtime agent)

## Tech Stack

- FastAPI + Uvicorn
- MySQL 8 + SQLAlchemy + Alembic
- Redis + RQ
- Qdrant (vector store, optional)
- MinIO (optional object storage)
- LiveKit (real-time audio sessions)
- OpenAI / DeepL (optional external providers)

## Services (Docker Compose)

- `mysql`: MySQL 8
- `redis`: Redis for cache/queues
- `qdrant`: vector database
- `minio`: optional object storage (profile: `with-storage`)
- `api`: FastAPI application
- `worker`: RQ worker (profile: `with-worker`, entrypoint not included in this repo)
- `livekit_realtime_agent`: LiveKit + OpenAI realtime worker

Note: `docker-compose.yml` also defines `livekit_sniffer` and `livekit_publisher` commands that expect
`workers/audio_sniffer.py` and `workers/publish_dual_outputs.py`. Those files are not in this repo.
Remove or comment out those services if you do not have the scripts.

## Quick Start (Docker)

### 1) Configure environment

```bash
cp .env.example .env
```

Set at minimum:
- `JWT_SECRET_KEY` (at least 32 chars)
- External provider keys (if not using mock)

### 2) Build and start services

```bash
make build
make up
```

### 3) Run migrations

```bash
make migrate
```

### 4) Optional seed (if scripts exist)

```bash
make seed
```

The seed target expects `scripts/seed_*.py`, which are not present in this repo.

### 5) Access

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

### 6) Stop services

```bash
make down
```

## Local Development (Poetry)

```bash
make install
```

Start infra locally (example):
```bash
docker-compose up -d mysql redis qdrant
```

Run the API and (optional) worker:
```bash
make run-local
make worker-local
```

The worker entrypoint referenced by `make worker-local` is not included in this repo.

## Configuration

Key environment variables (see `app/core/config.py` for full list):

- `API_PREFIX` (default empty; use `/api/v1` if you want a versioned prefix)
- `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`
- `DATABASE_URL` (or `MYSQL_*` when using compose)
- `REDIS_URL`
- `QDRANT_URL` / `QDRANT_HOST` / `QDRANT_PORT`
- `TRANSLATION_PROVIDER` = `MOCK` | `DEEPL` | `OPENAI`
- `OPENAI_API_KEY`, `DEEPL_API_KEY`
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `WORKER_SERVICE_KEY` (used by `/worker/token`)
- `CORS_ORIGINS`

## API Overview

Base URL is `http://localhost:8000` unless `API_PREFIX` is set.
Most endpoints require `Authorization: Bearer <JWT>`.

Auth:
- `POST /signup`
- `POST /general_login`

User, rooms, and friends:
- `GET /user/main`
- `GET /rooms/{room_id}`
- `POST /rooms/{room_id}/members`
- `POST /user/friend/add`

Meetings and sessions:
- `POST /meeting/{room_id}/live-sessions/{session_id}`
- `POST /meeting/{room_id}/live-sessions/{session_id}/leave`
- `GET /meeting/{session_id}/messages`
- `POST /meeting/livekit/token`

Translation:
- `POST /translation/translate`

Summary (mock responses):
- `POST /summary/{room_id}`
- `POST /summarization/{room_id}`
- `POST /meeting_member/{room_id}`
- `POST /translation_log/{room_id}`
- `POST /debug/summary/setup-mock`

Worker tokens:
- `POST /worker/token`

Debug:
- `GET /debug/user_info`
- `DELETE /debug/clear`

See Swagger for full details.

## WebSocket

Meeting session socket:
- `WS /meeting/{session_id}?token=<JWT>`

Server messages include `session_connected`, `pong`, and `unknown_type`.
Client messages include `chat` (requires auth) and `ping`.

## Background Workers

- RQ worker: `make worker` (Docker) or `make worker-local` (requires your own worker entrypoint)
- LiveKit realtime agent: `python workers/realtime_agent.py`

The realtime agent expects LiveKit + OpenAI Realtime environment variables
(see `.env.example` for defaults).

## Data Viewer (Streamlit)

There is a simple MySQL viewer at `app/dashboard/data_app.py`.

```bash
poetry run streamlit run app/dashboard/data_app.py
```

## Project Structure

```
URITOMO-Backend/
├── app/
│   ├── api/                 # REST API routers
│   ├── core/                # Config, auth, logging, errors
│   ├── debug/               # Debug endpoints
│   ├── infra/               # DB/Redis/Qdrant clients
│   ├── meeting/             # Rooms, sessions, LiveKit, websockets
│   ├── models/              # SQLAlchemy models
│   ├── summarization/       # Summary helpers
│   ├── translation/         # Translation services
│   ├── user/                # User/DM/room helpers
│   ├── worker/              # Worker token API
│   └── main.py              # FastAPI app entry
├── migrations/              # Alembic migrations
├── workers/                 # LiveKit/OpenAI realtime worker
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Make Commands

```bash
make help
make up
make up-storage
make down
make restart
make logs
make logs-api
make logs-worker
make ps
make clean
make build

make migrate
make migrate-create name=your_migration_name
make migrate-downgrade
make seed
make db-shell

make lint
make format

make shell
make shell-bash
make worker
make install
make run-local
make worker-local
make init
```

## Git Conventions

Branch naming: `[type]/[description]`

Examples: `feature/user-auth`, `fix/login-error`, `docs/api-guide`

Commit messages follow Conventional Commits:
`feat: add real-time translation websocket endpoint`

## License

MIT. See `LICENSE`.
