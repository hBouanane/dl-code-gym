# 08 - Docker Compose: API + Postgres with healthcheck, env, volume

```text
Write compose.yaml with two services (the Dockerfile and app are given):
  db:  image postgres:16-alpine ; environment POSTGRES_USER=gym POSTGRES_PASSWORD=gym POSTGRES_DB=gym
       volume pgdata:/var/lib/postgresql/data
       healthcheck: test ["CMD-SHELL", "pg_isready -U gym -d gym"], interval 5s, timeout 3s, retries 10
  api: build: . ; ports "8000:8000"
       environment DATABASE_URL=postgresql+psycopg://gym:gym@db:5432/gym   <- host = SERVICE NAME "db"
       depends_on: db: condition: service_healthy   (wait until Postgres really accepts connections)
       restart: unless-stopped
  volumes: pgdata:
Services on the same compose project share a network and resolve each other by service name.
```

## Files

- `main.py` (given)
- `requirements.txt` (given)
- `Dockerfile` (given)
- `compose.blank.yaml` (you write) / `compose.solution.yaml`

## Check (run in this folder)

```bash
docker compose -f compose.blank.yaml up -d --build
docker compose -f compose.blank.yaml ps          # both services running, db "healthy"
curl -X POST http://localhost:8000/visits        # {"total_visits":1}
curl -X POST http://localhost:8000/visits        # {"total_visits":2}
docker compose -f compose.blank.yaml down        # stop (data kept in the pgdata volume)
docker compose -f compose.blank.yaml up -d
curl -X POST http://localhost:8000/visits        # {"total_visits":3}  -> data survived
docker compose -f compose.blank.yaml down -v     # -v also deletes the volume
```
