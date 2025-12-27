# URITOMO Backend

Real-time translation service with cultural context and RAG-powered explanations for multilingual meetings.

## 🎯 Overview

URITOMO provides:
- **Real-time translation** via WebSocket with streaming support
- **Cultural context explanations** using RAG (Retrieval-Augmented Generation)
- **Meeting summaries** with action items and decisions
- **Organization glossaries** for domain-specific terminology
- **Hybrid explanation triggers** (rule-based + AI-powered)

## 🛠 Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: MySQL 8.0 + SQLAlchemy 2.0 + Alembic
- **Cache/Queue**: Redis + RQ
- **Vector DB**: Qdrant
- **Storage**: MinIO (optional)
- **AI**: OpenAI GPT-4 / DeepL (with mock mode for development)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### 1. Clone & Setup Environment

```bash
git clone <repository-url>
cd URITOMO-Backend

# Copy environment template
cp .env.example .env

# Edit .env and set your API keys (or use MOCK mode)
```

### 2. Start Services

```bash
# Initialize and start all services
make init

# Or manually:
make build
make up
make migrate
make seed
```

### 3. Verify Installation

```bash
# Check health
make health

# Or visit:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Qdrant Dashboard: http://localhost:6333/dashboard
```

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
│   │   ├── auth.py
│   │   ├── meetings.py
│   │   ├── segments.py
│   │   └── ws_realtime.py
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
5. Submit PR

## 📄 License

[Your License Here]

## 🔗 Links

- [API Documentation](http://localhost:8000/docs)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)