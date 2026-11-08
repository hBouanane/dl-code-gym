# 10 - ML image: PyTorch CPU inference API (small image, model at startup)

```text
serve.py is a FastAPI app that builds a small PyTorch model in its lifespan and serves /predict.
Dockerfile:
  - python:3.12-slim
  - install torch from the CPU-only wheel index (the default CUDA wheels are ~2.5 GB!):
      RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    then the rest: RUN pip install --no-cache-dir -r requirements.txt
    (separate layers: torch is huge and rarely changes -> cached)
  - copy serve.py, non-root user, EXPOSE 8000, CMD uvicorn serve:app ...
  - bonus: ENV OMP_NUM_THREADS=1 to keep CPU usage predictable when running many replicas
For a real model: COPY model.pt into the image, or mount it as a volume / download at startup.
```

## Files

- `serve.py` (given)
- `requirements.txt` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-10 .
docker images gym-10                     # about 1 GB with CPU wheels (the default CUDA wheels give 5+ GB)
docker run --rm -d --name gym-10 -p 8000:8000 gym-10
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"values\": [0.1, 0.2, 0.3, 0.4]}"
docker rm -f gym-10
```
