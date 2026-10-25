FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    SAVE_ENABLED=0 \
    STORE_DIR=/tmp/collected

WORKDIR /app

# dependencies first so this layer is cached between code changes
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# only what serving needs (see .dockerignore): app, feature code, clean pickle
COPY webapp/ webapp/
COPY public/ public/
COPY Model4/deploy/features.py Model4/deploy/features.py
COPY Model4/deploy/artifacts/clean/ Model4/deploy/artifacts/clean/

RUN useradd --create-home appuser && mkdir -p /tmp/collected && chown -R appuser /app /tmp/collected
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"

# 2 workers x 4 threads; each worker loads the 2.5 MB pipeline once
CMD ["gunicorn", "--chdir", "webapp", "-w", "2", "--threads", "4", "-b", "0.0.0.0:8080", "--timeout", "60", "--access-logfile", "-", "app:create_app()"]
