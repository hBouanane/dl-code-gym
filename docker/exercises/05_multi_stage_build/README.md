# 05 - Multi-stage build: build tools in one stage, slim runtime in another

```text
Smaller, safer images: compile/install in a "builder" stage, copy only the result.
  Stage 1 `builder` (FROM python:3.12-slim AS builder):
     RUN python -m venv /opt/venv ; ENV PATH="/opt/venv/bin:$PATH"
     COPY requirements.txt ; RUN pip install --no-cache-dir -r requirements.txt
     (this is where you'd apt-get install build-essential if a package needs compiling)
  Stage 2 `runtime` (FROM python:3.12-slim):
     COPY --from=builder /opt/venv /opt/venv ; ENV PATH="/opt/venv/bin:$PATH"
     non-root user, COPY main.py, CMD uvicorn ...
Compare sizes: docker images gym-05  vs an image with build-essential left inside.
```

## Files

- `main.py` (given)
- `requirements.txt` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-05 .
docker images gym-05
docker run --rm -d --name gym-05 -p 8000:8000 gym-05
curl http://localhost:8000/health
# expected: {"status":"ok","build":"multi-stage"}
docker rm -f gym-05
```
