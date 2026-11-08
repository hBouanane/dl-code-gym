# 06 - Build-time ARG vs run-time ENV, env files

```text
config.py prints APP_VERSION, GREETING and DEBUG from the environment.
  - ARG APP_VERSION=dev          -> only exists while BUILDING
  - ENV APP_VERSION=$APP_VERSION -> bake it into the image so the app sees it at runtime
  - ENV GREETING="hello" DEBUG="false" as defaults that can be overridden with -e or --env-file
Never put secrets in ARG/ENV of a Dockerfile: they stay visible in `docker history`.
```

## Files

- `config.py` (given)
- `prod.env` (given)
- `Dockerfile.blank` (you write) / `Dockerfile.solution`

## Check (run in this folder)

```bash
docker build -f Dockerfile.blank -t gym-06 --build-arg APP_VERSION=1.2.3 .
docker run --rm gym-06
# expected: version=1.2.3 greeting=hello debug=false
docker run --rm -e DEBUG=true gym-06
# expected: version=1.2.3 greeting=hello debug=true
docker run --rm --env-file prod.env gym-06
# expected: version=1.2.3 greeting=salam debug=false
```
