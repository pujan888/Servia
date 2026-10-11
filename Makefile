.PHONY: up db down logs migrate-docker seed-docker deploy-up deploy-down deploy-logs deploy-migrate install api agent jobs console migrate revision seed test lint

LOCAL = docker compose -f docker-compose.local.yml

# Local, in docker.
up:             ; $(LOCAL) up -d --build
db:             ; $(LOCAL) up -d postgres redis
down:           ; $(LOCAL) down
logs:           ; $(LOCAL) logs -f api agent jobs
migrate-docker: ; $(LOCAL) run --rm api alembic upgrade head
seed-docker:    ; $(LOCAL) run --rm api python -m app.seed

# On the server: docker-compose.yml, settings in server/.env.
deploy-up:      ; docker compose up -d --build
deploy-down:    ; docker compose down
deploy-logs:    ; docker compose logs -f api agent jobs
deploy-migrate: ; docker compose run --rm api alembic upgrade head

# Running on the host instead (needs `make db` first).
# The Python lives in server/; each command runs there.
install:  ; cd server && uv sync
api:      ; cd server && uv run uvicorn app.main:app --reload --port 8000
agent:    ; cd server && uv run python -m app.voice.worker dev
console:  ; cd server && uv run python -m app.voice.worker console
jobs:     ; cd server && uv run python -m app.jobs_worker
migrate:  ; cd server && uv run alembic upgrade head
revision: ; cd server && uv run alembic revision --autogenerate -m "$(m)"
seed:     ; cd server && uv run python -m app.seed
test:     ; cd server && uv run pytest -q
lint:     ; cd server && uv run ruff check app && uv run ruff format --check app
