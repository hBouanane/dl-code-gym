# 01 - Your first Dockerfile: FROM, WORKDIR, COPY, CMD

```text
Containerize hello.py (already in this folder).
  - base image: python:3.12-slim   (slim = small Debian image with Python)
  - working directory /app
  - copy hello.py into the image
  - default command: python hello.py   (use the JSON "exec form": CMD ["python", "hello.py"])
Know the difference: RUN = executed at BUILD time ; CMD = executed when the container STARTS.
```

## Files

- `hello.py` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-01 .
docker run --rm gym-01
# expected: Hello from Docker! Python 3.12.x
```
