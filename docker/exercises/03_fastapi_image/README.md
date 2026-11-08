# 03 - Containerize a FastAPI app (uvicorn, ports, ENV)

```text
Image for the FastAPI app in main.py.
  - python:3.12-slim, WORKDIR /app
  - ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1   (no .pyc files ; logs appear immediately)
  - install requirements first (cache!), then copy main.py
  - EXPOSE 8000   (documentation only, it doesn't publish the port)
  - CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    0.0.0.0 is REQUIRED: 127.0.0.1 inside a container is unreachable from your machine.
Run with -p 8000:8000 (host:container) and open http://localhost:8000/docs
```

## Files

- `main.py` (given)
- `requirements.txt` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-03 .
docker run -d --name gym-03 -p 8000:8000 -e APP_ENV=prod gym-03
curl http://localhost:8000/health
# expected: {"status":"ok","hostname":"<container id>","env":"prod"}
docker logs gym-03
docker rm -f gym-03
```
