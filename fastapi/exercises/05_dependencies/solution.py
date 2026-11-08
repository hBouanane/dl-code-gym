"""
Exercise 05 - Dependency injection with Depends (and overriding it in tests)  [di]

1. Function dependency: pagination(skip: int = 0, limit: int = Query(10, le=100)) -> dict
   GET /items uses `page: dict = Depends(pagination)` and returns ITEMS[skip: skip + limit]
2. Dependency with yield (setup/teardown, like a DB session):
     def get_db(): db = FakeDB(); try: yield db  finally: db.close()
   GET /db-status returns {"connected": db.connected}; after the request db.closed must be True
3. Class dependency: class Settings with a `greeting` attribute; def get_settings() -> Settings
   GET /greet uses settings: Settings = Depends(get_settings)
4. Annotated style (modern): SettingsDep = Annotated[Settings, Depends(get_settings)]
5. Header-based dependency verify_token(x_token: str = Header()) that raises 401 unless
   x_token == "secret"; protect GET /private with dependencies=[Depends(verify_token)]
6. In the test: app.dependency_overrides[get_settings] = lambda: Settings(greeting="Test hi")
Self-check expects these names: app, get_settings, Settings, CREATED_DBS (list of FakeDB created)
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query

app = FastAPI()
ITEMS = [f"item-{i}" for i in range(50)]
CREATED_DBS = []


def pagination(skip: int = 0, limit: int = Query(10, le=100)) -> dict:
    return {"skip": skip, "limit": limit}


class FakeDB:
    def __init__(self):
        self.connected, self.closed = True, False
        CREATED_DBS.append(self)

    def close(self):
        self.connected, self.closed = False, True


def get_db():
    db = FakeDB()
    try:
        yield db
    finally:
        db.close()


class Settings:
    def __init__(self, greeting: str = "Hello"):
        self.greeting = greeting


def get_settings() -> Settings:
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def verify_token(x_token: str = Header()):
    if x_token != "secret":
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/items")
def list_items(page: dict = Depends(pagination)):
    return ITEMS[page["skip"]: page["skip"] + page["limit"]]


@app.get("/db-status")
def db_status(db: FakeDB = Depends(get_db)):
    return {"connected": db.connected}


@app.get("/greet")
def greet(settings: SettingsDep):
    return {"message": settings.greeting}


@app.get("/private", dependencies=[Depends(verify_token)])
def private():
    return {"secret": "data"}


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    c = TestClient(app)
    assert c.get("/items").json() == [f"item-{i}" for i in range(10)]
    assert c.get("/items", params={"skip": 45, "limit": 10}).json() == [f"item-{i}" for i in range(45, 50)]
    assert c.get("/items", params={"limit": 500}).status_code == 422
    assert c.get("/db-status").json() == {"connected": True} and CREATED_DBS[-1].closed
    assert c.get("/greet").json() == {"message": "Hello"}
    app.dependency_overrides[get_settings] = lambda: Settings(greeting="Test hi")
    assert c.get("/greet").json() == {"message": "Test hi"}
    app.dependency_overrides.clear()
    assert c.get("/private").status_code == 422
    assert c.get("/private", headers={"X-Token": "wrong"}).status_code == 401
    assert c.get("/private", headers={"X-Token": "secret"}).json() == {"secret": "data"}
    print("All checks passed!")
