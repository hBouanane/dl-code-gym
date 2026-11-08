"""
Exercise 10 - Lifespan (startup/shutdown) and BackgroundTasks  [middleware]

1. @asynccontextmanager async def lifespan(app):
       app.state.model = load_model()     <- startup: load an ML model / DB pool ONCE
       EVENTS.append("startup")
       yield
       EVENTS.append("shutdown")          <- shutdown: close connections
   app = FastAPI(lifespan=lifespan)
   load_model() returns a fake "model": lambda text: len(text)
2. POST /predict {"text": ...} uses request.app.state.model -> {"score": ...}
3. BackgroundTasks: POST /signup?email=... -> returns immediately {"status": "ok"} and
   schedules background_tasks.add_task(send_welcome_email, email) which appends to SENT_EMAILS
4. Test: `with TestClient(app) as c:` (the `with` is what triggers startup/shutdown!)
Self-check expects these names: app, EVENTS, SENT_EMAILS
"""

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from pydantic import BaseModel

EVENTS: list[str] = []
SENT_EMAILS: list[str] = []


def load_model():
    return lambda text: len(text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    EVENTS.append("startup")
    yield
    EVENTS.append("shutdown")


app = FastAPI(lifespan=lifespan)


class PredictIn(BaseModel):
    text: str


@app.post("/predict")
def predict(body: PredictIn, request: Request):
    return {"score": request.app.state.model(body.text)}


def send_welcome_email(email: str):
    SENT_EMAILS.append(email)


@app.post("/signup")
def signup(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_welcome_email, email)
    return {"status": "ok"}


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        assert EVENTS[-1] == "startup"
        assert c.post("/predict", json={"text": "hello"}).json() == {"score": 5}
        assert c.post("/signup", params={"email": "a@b.com"}).json() == {"status": "ok"}
        assert SENT_EMAILS[-1] == "a@b.com"
    assert EVENTS[-1] == "shutdown"
    print("All checks passed!")
