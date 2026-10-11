FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/srv/.venv/bin:$PATH"

WORKDIR /srv

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Bake the turn-detector and VAD model files into the image. Settings are required at import
# time, so placeholders stand in; the real values come from .env at runtime.
RUN DATABASE__URL=postgresql+asyncpg://build@localhost/build \
    LIVEKIT__URL=ws://localhost LIVEKIT__API_KEY=build LIVEKIT__API_SECRET=build \
    python -m app.voice.worker download-files

EXPOSE 8000
# migrations run first, then the CMD below
ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
