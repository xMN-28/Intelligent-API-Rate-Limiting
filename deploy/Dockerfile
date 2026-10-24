FROM python:3.12.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY configs ./configs
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --create-home app && mkdir -p /app/results/runs && chown -R app:app /app
USER app
CMD ["irl", "serve", "gateway", "--host", "0.0.0.0"]
