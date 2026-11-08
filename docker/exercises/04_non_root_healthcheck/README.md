# 04 - Security & ops: non-root USER and HEALTHCHECK

```text
Same app as 03, production-hardened:
  - create a user: RUN useradd --create-home --uid 1000 appuser   and switch to it with USER appuser
    (containers run as root by default; a compromised app would then be root inside the container)
  - copy files with --chown=appuser:appuser
  - HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD python -c
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
    (slim images have no curl, so use Python)
  - add a LABEL org.opencontainers.image.title="gym-api"
```

## Files

- `main.py` (given)
- `requirements.txt` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-04 .
docker run -d --name gym-04 -p 8000:8000 gym-04
docker exec gym-04 whoami
# expected: appuser
# wait ~15s, then:
docker inspect --format "{{.State.Health.Status}}" gym-04
# expected: healthy
docker rm -f gym-04
```
