# URITOMO Backend

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi) ![LiveKit](https://img.shields.io/badge/LiveKit-Realtime-blueviolet) ![Docs](https://img.shields.io/badge/API%20Docs-Swagger-green?logo=swagger) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow) ![Deploy](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker)

## Overview
- FastAPI + LiveKit backend for real-time JP↔KR meetings: auth, rooms, chat, STT translation, glossary, and ops dashboard in one place.

## Features
- JWT auth, room/member management, friends & DM
- LiveKit token issuance + Realtime Agent (STT → translate → Redis broadcast)
- WebSocket chat/STT events, REST translation & mock summarization
- Streamlit dashboard for quick table insights
- Worker token/service-key guardrails to protect ops workers

## System Architecture
```mermaid
graph LR
  C[Client Web/RTC] -->|HTTP/WS| API[FastAPI]
  API --> DB[(MySQL 8)]
  API --> Cache[(Redis 7)]
  API -->|LiveKit Token| LK[LiveKit]
  LK --> Agent[Realtime Agent]
  Agent -->|Events| Cache
  API --> Dash[Streamlit Dashboard]
  Agent -->|STT+Translate| LLM[OpenAI & DeepL]
```
## Tech Stack
| Area | Tools |
| --- | --- |
| Backend | FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2, Alembic |
| Data | MySQL 8, Redis 7 |
| Realtime/AI | LiveKit API/RTC, OpenAI SDK, DeepL |
| Infra | Docker & docker-compose, Makefile |
| Observability | structlog, python-json-logger |
| DevEx | Poetry, Ruff, Black, Mypy, Pytest |

## Implementation & Run Guide

### Setup / Deployment
- Prereqs: Docker, Docker Compose, Make. Prepare `.env`, then:
```bash
cp .env.example .env
make build && make up         # api + mysql + redis + dashboard
make migrate                  # Alembic upgrade head
./run.sh                      # build -> up -> migrate -> recreate LiveKit worker
# stop/clean: make down / make clean
```

- Compose profile: add `--profile with-worker` to include the worker container when available.
