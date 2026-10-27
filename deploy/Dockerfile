FROM python:3.12.10-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
ARG WITH_ML=false
RUN if [ "$WITH_ML" = "true" ]; then uv sync --frozen --no-dev --extra ml; else uv sync --frozen --no-dev; fi
COPY configs ./configs
COPY assets ./assets
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --create-home app && mkdir -p /app/results/runs && chown -R app:app /app
USER app
CMD ["rlrc", "serve", "gateway", "--host", "0.0.0.0"]
