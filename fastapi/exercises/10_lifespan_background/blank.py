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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





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
