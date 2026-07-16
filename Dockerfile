# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Builder
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# Set python path
RUN --mount=type=secret,id=guardrails_key \
    export GUARDRAILS_NO_PROMPT=true && \
    export GUARDRAILS_API_KEY=$(cat /run/secrets/guardrails_key) && \
    yes "n" | uv run guardrails configure --token $GUARDRAILS_API_KEY && \
    yes "y" | uv run guardrails hub install hub://guardrails/detect_pii --quiet && \
    yes "y" | uv run guardrails hub install hub://guardrails/gibberish_text --quiet && \
    yes "y" | uv run guardrails hub install hub://guardrails/toxic_language --quiet

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Runtime
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Create a non-root user for security
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser && \
    chown -R appuser:appgroup /app

# Copy only the virtual environment from builder (no uv, no build tools)
COPY --from=builder --chown=appuser:appgroup /app/.guardrails ./.guardrails
COPY --from=builder --chown=appuser:appgroup /app/.venv ./.venv
COPY --from=builder --chown=appuser:appgroup /app/src ./src
COPY --from=builder --chown=appuser:appgroup /app/scripts ./scripts
COPY --from=builder --chown=appuser:appgroup /app/data ./data
COPY --from=builder --chown=appuser:appgroup /app/secrets ./secrets

USER appuser

ENV PYTHONPATH=/app

EXPOSE 8000

# Health check using built-in Python to avoid installing curl/wget
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
