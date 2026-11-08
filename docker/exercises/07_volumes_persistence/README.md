# 07 - Volumes and bind mounts: data that survives the container

```text
counter.py increments a number stored in /data/count.txt.
Dockerfile: python:3.12-slim, copy counter.py, declare VOLUME /data, CMD python counter.py
Then learn the 3 storage types (commands in the check section):
  - no volume       -> the count resets every run (container filesystem is thrown away with --rm)
  - named volume    -> docker run -v gym-counter:/data ...  -> 1, 2, 3 ...  (managed by Docker)
  - bind mount      -> docker run -v "$(pwd)/data:/data" ... -> the file appears in YOUR folder
```

## Files

- `counter.py` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-07 .
docker run --rm gym-07                         # run number 1
docker run --rm gym-07                         # run number 1 again (no persistence)
docker run --rm -v gym-counter:/data gym-07    # run number 1
docker run --rm -v gym-counter:/data gym-07    # run number 2
docker run --rm -v "$(pwd)/data:/data" gym-07  # creates ./data/count.txt on your machine
docker volume ls
docker volume rm gym-counter
```
