# 09 - Dev workflow: compose override with bind mount + hot reload

```text
compose.yaml (given) is the production-like setup. Write compose.dev.yaml that OVERRIDES it for development:
  api:
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    volumes: - ./:/app        (bind mount your source code -> edits are visible instantly)
    environment: APP_ENV: dev
Compose merges files left to right: docker compose -f compose.yaml -f compose.dev.yaml up
Edit main.py while it runs -> uvicorn reloads, no rebuild needed.
```

## Files

- `main.py` (given)
- `requirements.txt` (given)
- `Dockerfile` (given)
- `compose.yaml` (given)
- `compose.dev.blank.yaml` (you write) / `compose.dev.solution.yaml`

## Check (run in this folder)

```bash
docker compose -f compose.yaml -f compose.dev.blank.yaml up --build
curl http://localhost:8000/        # {"message":"edit me and watch it reload","env":"dev"}
# edit the message in main.py, save, curl again -> new message without rebuilding
docker compose -f compose.yaml -f compose.dev.blank.yaml config   # shows the merged result
docker compose -f compose.yaml -f compose.dev.blank.yaml down
```
