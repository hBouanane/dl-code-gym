"""
Exercise 09 - Middleware: timing header, request id, CORS  [middleware]

1. @app.middleware("http") async def add_process_time(request, call_next):
       start = time.perf_counter(); response = await call_next(request)
       response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
2. A second middleware that reads "X-Request-ID" from the request (or generates uuid4().hex),
   stores it in request.state.request_id and echoes it back in the response header.
3. app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                      allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
4. GET /ping returns {"request_id": request.state.request_id}
Test CORS: a request with header Origin: http://localhost:3000 gets
"access-control-allow-origin" in the response; Origin: http://evil.com does not.
Self-check expects these names: app
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/ping")
def ping(request: Request):
    return {"request_id": request.state.request_id}


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/ping", headers={"X-Request-ID": "abc123"})
    assert r.json() == {"request_id": "abc123"} and r.headers["X-Request-ID"] == "abc123"
    assert float(r.headers["X-Process-Time"]) >= 0
    assert len(c.get("/ping").headers["X-Request-ID"]) == 32
    assert c.get("/ping", headers={"Origin": "http://localhost:3000"}).headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in c.get("/ping", headers={"Origin": "http://evil.com"}).headers
    print("All checks passed!")
