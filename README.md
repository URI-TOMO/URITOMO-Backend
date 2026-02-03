# URITOMO Backend

실시간 다국어 미팅을 위한 백엔드 API 서버입니다. LiveKit 기반 세션과 STT 번역/요약 파이프라인을 지원합니다.

## 빠른 시작 (Docker)

```bash
cp .env.example .env
```

`.env`에 최소한 다음을 설정하세요.

- `JWT_SECRET_KEY` (32자 이상)
- 번역/설명 기능을 쓸 경우 `DEEPL_API_KEY` 또는 `OPENAI_API_KEY`
- LiveKit 연동 시 `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- 워커 토큰 발급을 사용할 경우 `WORKER_SERVICE_KEY`

```bash
make build
make up
make migrate
```

API 문서:

```text
http://localhost:8000/docs
```

단일 스크립트 실행(빌드 + 마이그레이션 포함):

```bash
./run.sh
```

## 주요 기능

- JWT 인증 기반 사용자/프로필 관리
- 방/미팅/세션 라이프사이클 관리
- WebSocket 실시간 채팅 및 STT 이벤트 처리
- STT 번역(DeepL/Mock) 및 용어 설명(OpenAI)
- 요약 API(현재 mock 응답)
- LiveKit 토큰 발급 및 realtime agent 연동
- Redis 큐, Qdrant 연동, Streamlit 대시보드

## 기술 스택

- FastAPI, Uvicorn
- MySQL 8, SQLAlchemy, Alembic
- Redis, RQ
- Qdrant (optional)
- MinIO (optional)
- LiveKit
- OpenAI / DeepL (optional)
- Python 3.11, Poetry

## 서비스 구성 (docker-compose)

- `mysql`: MySQL 8
- `redis`: Redis
- `qdrant`: 벡터 DB
- `minio`: 오브젝트 스토리지 (profile: `with-storage`)
- `api`: FastAPI 서버
- `dashboard`: Streamlit 데이터 뷰어
- `worker`: 백그라운드 워커 (profile: `with-worker`)
- `livekit_realtime_agent`: LiveKit + OpenAI Realtime 워커

Note:
- `worker` 서비스는 `python -m app.workers.worker`를 실행하도록 되어 있습니다. 현재 레포에는 해당 모듈이 없으므로 필요 시 경로를 수정하거나 구현을 추가하세요.

## 환경 변수

설정 기준은 `app/core/config.py`입니다. `.env.example`를 참고하되, 아래 항목이 실제로 사용됩니다.

필수/중요:
- `JWT_SECRET_KEY` (필수, 32자 이상)
- `DATABASE_URL` (권장) 또는 `MYSQL_*` 조합
- `REDIS_URL`
- `API_PREFIX` (기본값 빈 문자열, 예: `/api/v1`)

선택:
- `TRANSLATION_PROVIDER` = `MOCK` | `DEEPL` | `OPENAI`
- `DEEPL_API_KEY`, `OPENAI_API_KEY`
- `QDRANT_URL` 또는 `QDRANT_HOST`/`QDRANT_PORT`
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `WORKER_SERVICE_KEY`

주의:
- `.env.example`의 `SECRET_KEY`, `API_V1_STR`는 앱 설정과 키 이름이 다릅니다. 실제로는 `JWT_SECRET_KEY`, `API_PREFIX`를 사용합니다.

## 실행 방법

### Docker

```bash
make up
```

MinIO 포함:

```bash
make up-storage
```

Worker 포함:

```bash
docker-compose --profile with-worker up -d
```

로그:

```bash
make logs
make logs-api
```

중지:

```bash
make down
```

### 로컬 개발 (Poetry)

```bash
make install
```

인프라만 실행:

```bash
docker-compose up -d mysql redis qdrant
```

API 실행:

```bash
make run-local
```

## DB 마이그레이션

```bash
make migrate
```

새 마이그레이션 생성:

```bash
make migrate-create name=add_new_table
```

## API & WebSocket

- 문서: `/docs`, `/redoc`
- 기본 경로는 `API_PREFIX`에 따라 달라집니다.

주요 엔드포인트(요약):
- 인증: `POST /signup`, `POST /general_login`
- 방/미팅: `POST /rooms`, `GET /rooms/{id}`, `POST /meeting/...`
- 번역: `POST /translation/stt`, `GET /translation/description/{room_id}`
- 요약: `POST /summarization/{room_id}` (mock)
- 워커 토큰: `POST /worker/token` (헤더 `X-Worker-Key` 필요)

WebSocket:
- 미팅: `WS /meeting/{room_id}?token=<JWT>`
- DM: `WS /dm/ws/{thread_id}?token=<JWT>`

## Realtime Agent

`workers/realtime_agent.py`는 LiveKit + OpenAI Realtime을 이용해 실시간 처리를 수행합니다. 필요한 환경 변수는 `.env.example`의 LiveKit/Realtime 항목을 참고하세요.

## 대시보드

Streamlit 대시보드는 8501 포트에서 제공됩니다. API 서버의 `/dashboard` 경로로 리다이렉트됩니다.

## 테스트/품질

- Lint: `make lint`
- Format: `make format`
- Test: `pytest`

## 프로젝트 구조

```
URITOMO-Backend/
├── app/
│   ├── api/             # REST API 라우터
│   ├── core/            # 설정, 보안, 로깅
│   ├── debug/           # 디버그 API
│   ├── infra/           # DB/Redis/Qdrant
│   ├── meeting/         # 미팅, WS, LiveKit
│   ├── models/          # SQLAlchemy 모델
│   ├── summarization/   # 요약 로직
│   ├── translation/     # 번역/설명 로직
│   ├── user/            # 사용자/친구/방
│   ├── worker/          # 워커 토큰 API
│   └── main.py          # FastAPI 엔트리
├── migrations/
├── workers/             # LiveKit realtime agent
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── run.sh
```

## 운영/배포 가이드 (요약)

- `DEBUG=false`, `ENV=production` 설정
- `JWT_SECRET_KEY`/외부 API 키는 안전하게 관리
- 프로덕션에서는 `--reload` 제거 및 적절한 ASGI 서버 구성

## 라이선스

`LICENSE` 참고
