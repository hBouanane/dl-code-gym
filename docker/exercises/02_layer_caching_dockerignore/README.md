# 02 - Layer caching + .dockerignore

```text
app.py uses the `pyfiglet` package (listed in requirements.txt).
1. Write the Dockerfile so that changing app.py does NOT reinstall the dependencies:
     COPY requirements.txt FIRST -> RUN pip install --no-cache-dir -r requirements.txt -> THEN COPY the code
   (each instruction is a cached layer; a layer is rebuilt only if it or a previous one changed)
2. Write .dockerignore to keep junk out of the build context:
     __pycache__/, *.pyc, .venv/, .git/, .env, *.log, data/
Experiment: build, edit the message in app.py, build again -> the pip step says CACHED.
```

## Files

- `app.py` (given)
- `requirements.txt` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`
- `.dockerignore.blank` (you write) / `.dockerignore.solution`

## Check (run in this folder)

```bash
# (rename .dockerignore.blank to .dockerignore when you're done writing it: Docker only reads that exact name)
docker build -f Dockerfile.blank -t gym-02 .
docker run --rm gym-02
# edit app.py, rebuild: the "RUN pip install" step must show CACHED
docker build -f Dockerfile.blank -t gym-02 .
```
