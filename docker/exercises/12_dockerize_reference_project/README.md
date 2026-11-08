# 12 - Dockerize the FastAPI reference project (Dockerfile + compose)

```text
Context: ../../../fastapi/reference_project (the hexagonal Library API).
Write, from memory, a Dockerfile for it that:
  - uses python:3.12-slim, the ENV trio (PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED, PIP_NO_CACHE_DIR)
  - installs requirements.txt before copying the app/ package (cache)
  - does NOT copy tests/ (use the project's .dockerignore)
  - runs as a non-root user owning /data
  - sets LIB_DATABASE_URL=sqlite:////data/library.db (4 slashes = absolute path)
  - has a HEALTHCHECK on /health, EXPOSE 8000, CMD uvicorn app.main:app on 0.0.0.0
Then compare with ../../../fastapi/reference_project/Dockerfile and its compose.yaml.
```

## Files


- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
cd ../../../fastapi/reference_project
docker build -f ../../docker/exercises/12_dockerize_reference_project/Dockerfile.blank -t library-api .
docker run --rm -d --name library -p 8000:8000 -v library-data:/data library-api
curl -X POST http://localhost:8000/books -H "Content-Type: application/json" -d "{\"title\": \"Dune\", \"author\": \"Herbert\"}"
curl http://localhost:8000/books
docker rm -f library
```
