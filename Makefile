.PHONY: help up down restart logs ps clean build migrate migrate-create seed test lint format shell db-shell worker

# Default target
help:
	@echo "URITOMO Backend - Available Commands:"
	@echo ""
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - View logs from all services"
	@echo "  make logs-api        - View API logs only"
	@echo "  make logs-worker     - View worker logs only"
	@echo "  make ps              - List running containers"
	@echo "  make clean           - Remove all containers and volumes"
	@echo "  make build           - Rebuild all images"
	@echo ""
	@echo "  make migrate         - Run database migrations"
	@echo "  make migrate-create  - Create a new migration (name=migration_name)"
	@echo "  make seed            - Seed database with sample data"
	@echo "  make db-shell        - Open MySQL shell"
	@echo ""
	@echo "  make lint            - Run linters"
	@echo "  make format          - Format code with black and isort"
	@echo ""
	@echo "  make shell           - Open Python shell in API container"
	@echo "  make worker          - Start background worker"
	@echo "  make install         - Install dependencies locally"

# Docker Compose commands
up:
	docker-compose up -d
	@echo "Services started. API available at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-worker:
	docker-compose logs -f worker

ps:
	docker-compose ps

clean:
	docker-compose down -v
	@echo "All containers and volumes removed."

build:
	docker-compose build --no-cache

# Database commands
migrate:
	docker-compose exec api alembic upgrade head

migrate-create:
	@if [ -z "$(name)" ]; then \
		echo "Error: Please provide a migration name: make migrate-create name=your_migration_name"; \
		exit 1; \
	fi
	docker-compose exec api alembic revision --autogenerate -m "$(name)"

migrate-downgrade:
	docker-compose exec api alembic downgrade -1

seed:
	docker-compose exec api python scripts/seed_culture_cards.py
	docker-compose exec api python scripts/seed_glossary.py
	@echo "Sample data seeded successfully."

db-shell:
	docker-compose exec mysql mysql -u uritomo_user -p uritomo

# Quality
lint:
	docker-compose exec api ruff check app/
	docker-compose exec api mypy app/

format:
	docker-compose exec api black app/
	docker-compose exec api isort app/
	docker-compose exec api ruff check --fix app/

# Development helpers
shell:
	docker-compose exec api python

shell-bash:
	docker-compose exec api bash

worker:
	docker-compose exec worker python -m app.workers.worker

# Local development (without Docker)
install:
	poetry install

run-local:
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker-local:
	poetry run python -m app.workers.worker

# Initial setup
init: build up migrate seed
	@echo ""
	@echo "✅ Initial setup complete!"
	@echo "API is running at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"
	@echo ""

# Health check
health:
	@curl -f http://localhost:8000/api/v1/health || echo "API is not responding"
