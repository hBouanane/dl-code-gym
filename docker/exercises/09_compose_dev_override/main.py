import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "edit me and watch it reload", "env": os.getenv("APP_ENV", "prod")}
