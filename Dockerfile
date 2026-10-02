FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /code

COPY pyproject.toml uv.lock README.md ./
# The locked dependencies include the spaCy model wheel. No model download is
# needed when the container starts or when a request arrives.
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app
RUN useradd --create-home --uid 10001 api
USER api
EXPOSE 80
CMD ["/code/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
