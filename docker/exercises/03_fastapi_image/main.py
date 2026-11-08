import os
import socket

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "hostname": socket.gethostname(), "env": os.getenv("APP_ENV", "dev")}
