# URITOMO Backend

Real-time translation service with cultural context and RAG-powered explanations for multilingual meetings.

## 🎯 Overview

URITOMO provides:
- **Real-time translation** via WebSocket with streaming support
- **Cultural context explanations** using RAG (Retrieval-Augmented Generation)
- **Meeting summaries** with action items and decisions
- **Organization glossaries** for domain-specific terminology
- **Hybrid explanation triggers** (rule-based + AI-powered)

## 🚀 Quick Start (Docker)

If you already have Docker installed, follow these steps to quickly run the services.

### 1. Environment Configuration (.env)
Copy the example environment file to create your own.

```bash
cp .env.example .env
```

Open the `.env` file and configure your external API keys (OpenAI, DeepL, etc.). Even without API keys, you can perform local testing by setting `TRANSLATION_PROVIDER=MOCK`.

### 2. Start Services

You can use the `Makefile` to complete all configurations at once:

```bash
# Performs initial build, execution, DB migrations, and sample data seeding.
make init
```

Or to use Docker commands directly:

```bash
# Build and start containers
docker-compose up -d --build

# Create DB tables
docker-compose exec api alembic upgrade head

# Insert sample data (cultural guides, etc.)
docker-compose exec api python scripts/seed_culture_cards.py
```

### 3. Verification & Access

Once services are running, you can access them at:
- **FastAPI Server**: [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant (Vector DB)**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

To check logs while running:
```bash
make logs
# or
docker-compose logs -f
```

### 4. Stopping the Server
To stop all running services:

**Mac / Linux:**
```bash
make down
```

**Windows:**
```powershell
docker-compose down
```

---
## 💻 Running on Windows

Since `make` is not natively available on Windows, you have a few options:

### Option 1: WSL2 (Recommended)
If you have **WSL2 (Windows Subsystem for Linux)** installed:
1. Open your WSL terminal (e.g., Ubuntu).
2. Follow the Mac/Linux instructions exactly (`make init`).

### Option 2: Manual Commands (PowerShell / CMD)
If you don't have `make`, run these commands in order:

1. **Build & Start**:
   ```powershell
   docker-compose build --no-cache
   docker-compose up -d
   ```

2. **Run Migrations**:
   ```powershell
   docker-compose exec api alembic upgrade head
   ```

3. **Seed Data**:
   ```powershell
   docker-compose exec api python scripts/seed_culture_cards.py
   docker-compose exec api python scripts/seed_glossary.py
   ```

---

## 👨‍💻 Git Conventions

### Branch Naming
Follow this format: `[type]/[description]`

| Type | Description | Example |
|------|-------------|---------|
| `feature` | New features | `feature/user-auth` |
| `fix` | Bug fixes | `fix/login-error` |
| `hotfix` | Critical production fixes | `hotfix/security-patch` |
| `chore` | Maintenance, config changes | `chore/update-dependencies` |
| `docs` | Documentation updates | `docs/api-guide` |
| `refactor` | Code restructuring | `refactor/segment-logic` |

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

**Format**: `[Type]: [Description]`

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries

**Examples**:
- `feat: add real-time translation websocket endpoint`
- `fix: resolve crash when Qdrant is unavailable`
- `chore: update poetry dependencies`

---

## 🛠 Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: MySQL 8.0 + SQLAlchemy 2.0 + Alembic
- **Cache/Queue**: Redis + RQ
- **Vector DB**: Qdrant
- **Storage**: MinIO (optional)
- **AI**: OpenAI GPT-4 / DeepL (with mock mode for development)

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### REST API
```
POST   /api/v1/auth/register          - Register new user
POST   /api/v1/auth/login             - Login and get JWT token
GET    /api/v1/orgs                   - List organizations
POST   /api/v1/meetings               - Create meeting
POST   /api/v1/segments               - Ingest transcript segment
POST   /api/v1/meetings/{id}/summary  - Trigger meeting summary
GET    /api/v1/meetings/{id}/summary  - Get meeting summary
```

#### WebSocket
```
WS     /api/v1/ws/realtime?token=<JWT>&meeting_id=<ID>
```

**Client → Server messages:**
```json
{
  "type": "segment.ingest",
  "data": {
    "meeting_id": "uuid",
    "speaker": "John",
    "lang": "ja",
    "text": "検討します",
    "ts": 1234567890
  }
}
```

**Server → Client messages:**
```json
{
  "type": "translation.final",
  "data": {
    "segment_id": "uuid",
    "translated_text": "검토하겠습니다",
    "explanation_text": "일본 비즈니스 문화에서 '검토합니다'는...",
    "confidence": 0.95
  }
}
```

## 🗂 Project Structure

```
URITOMO-Backend/
├── app/
│   ├── main.py                 # FastAPI application entry
│   ├── core/                   # Core configurations
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── logging.py
│   │   └── deps.py
│   ├── api/v1/                 # API endpoints
│   │   ├── core/               # Core services (Auth, Orgs, Meetings)
│   │   │   ├── auth.py
│   │   │   ├── health.py
│   │   │   ├── meetings.py
│   │   │   └── orgs.py
│   │   ├── ai/                 # AI features (Segments, Realtime)
│   │   │   ├── segments.py
│   │   │   └── ws_realtime.py
│   │   └── examples/           # Verification examples
│   ├── models/                 # SQLAlchemy models
│   ├── schemas/                # Pydantic schemas
│   ├── services/               # Business logic
│   │   ├── translation_service.py
│   │   ├── explanation_service.py
│   │   ├── summary_service.py
│   │   ├── rag_service.py
│   │   └── llm_clients/
│   ├── infra/                  # Infrastructure
│   │   ├── db.py
│   │   ├── redis.py
│   │   ├── qdrant.py
│   │   └── queue.py
│   ├── workers/                # Background jobs
│   │   └── jobs/
│   └── prompts/                # LLM prompts
├── migrations/                 # Alembic migrations
├── scripts/                    # Utility scripts
├── tests/                      # Test suite
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## 🧪 Development

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# WebSocket tests only
make test-ws
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint
```

### Database Migrations

```bash
# Create new migration
make migrate-create name=add_new_field

# Apply migrations
make migrate

# Rollback last migration
make migrate-downgrade
```

### Background Worker

```bash
# View worker logs
make logs-worker

# Restart worker
docker-compose restart worker
```

## 🔧 Configuration

### Mock Mode (No API Keys Required)

Set in `.env`:
```bash
TRANSLATION_PROVIDER=MOCK
EMBEDDING_PROVIDER=MOCK
SUMMARY_PROVIDER=MOCK
```

### Production Mode

Set in `.env`:
```bash
TRANSLATION_PROVIDER=OPENAI  # or DEEPL
OPENAI_API_KEY=sk-...
DEEPL_API_KEY=...
EMBEDDING_PROVIDER=OPENAI
```

## 📝 Available Make Commands

```bash
make help              # Show all commands
make up                # Start services
make down              # Stop services
make logs              # View all logs
make migrate           # Run migrations
make seed              # Seed sample data
make test              # Run tests
make clean             # Clean all containers & volumes
```

## 🌐 WebSocket Protocol

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/realtime?token=YOUR_JWT&meeting_id=MEETING_ID');
```

### Message Types

**Client → Server:**
- `segment.ingest`: Send new transcript segment
- `settings.update`: Update translation settings

**Server → Client:**
- `segment.ack`: Acknowledgment
- `translation.partial`: Streaming translation chunk
- `translation.final`: Complete translation with explanation
- `error`: Error message

## 🎓 RAG & Cultural Cards

The system includes 50+ pre-seeded cultural cards for Japanese business expressions:

- "検討します" → Often means "no" in polite form
- "頑張ります" → Commitment expression, context matters
- "よろしくお願いします" → Multi-purpose greeting/request

Customize with your own cards using `scripts/seed_culture_cards.py`.

## 📊 Monitoring

- **Logs**: `make logs` or `make logs-api`
- **Health**: `curl http://localhost:8000/api/v1/health`
- **Metrics**: (Coming soon: Prometheus integration)

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run `make format` and `make lint`
4. Run `make test`
5. Submit PR to the `dev` branch (Do NOT merge directly into `main`)

## 📄 License

[Your License Here]

## 🔗 Links

- [API Documentation](http://localhost:8000/docs)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)