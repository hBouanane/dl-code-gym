"""
Exercise 14 - Layered architecture: router -> service -> repository  [architecture]

The first step away from "everything in the route function". Write 3 layers (one file here;
in a project they would be api/, services/, repositories/):
  REPOSITORY (data access only, no business rules):
     class UserRepository: add(user: dict) -> dict ; get_by_email(email) -> dict | None ; list() -> list
  SERVICE (business rules only, no HTTP, no SQL):
     class UserService(repo): register(name, email) -> dict
        - email is lowercased
        - raises EmailAlreadyUsed (a plain Python exception) if the email exists
  API (HTTP only: parse request, call the service, map errors to status codes):
     get_user_service() dependency returning UserService(repo) with a module-level repo
     POST /users -> 201 ; EmailAlreadyUsed -> 409
     GET /users -> list
Rule to feel: the service can be tested with NO FastAPI at all.
Self-check expects these names: app, UserRepository, UserService, EmailAlreadyUsed, get_user_service
"""

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel


# ---------- repository layer ----------
class UserRepository:
    def __init__(self):
        self._users: list[dict] = []

    def add(self, user: dict) -> dict:
        user = {"id": len(self._users) + 1, **user}
        self._users.append(user)
        return user

    def get_by_email(self, email: str) -> dict | None:
        return next((u for u in self._users if u["email"] == email), None)

    def list(self) -> list[dict]:
        return list(self._users)


# ---------- service layer ----------
class EmailAlreadyUsed(Exception):
    pass


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def register(self, name: str, email: str) -> dict:
        email = email.lower()
        if self.repo.get_by_email(email):
            raise EmailAlreadyUsed(email)
        return self.repo.add({"name": name, "email": email})

    def list_users(self) -> list[dict]:
        return self.repo.list()


# ---------- api layer ----------
class UserIn(BaseModel):
    name: str
    email: str


_repo = UserRepository()


def get_user_service() -> UserService:
    return UserService(_repo)


app = FastAPI()


@app.post("/users", status_code=status.HTTP_201_CREATED)
def register(body: UserIn, service: UserService = Depends(get_user_service)):
    try:
        return service.register(body.name, body.email)
    except EmailAlreadyUsed:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already used")


@app.get("/users")
def list_users(service: UserService = Depends(get_user_service)):
    return service.list_users()


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    svc = UserService(UserRepository())
    assert svc.register("A", "A@X.com")["email"] == "a@x.com"
    try:
        svc.register("B", "a@x.com")
        raise AssertionError("should raise")
    except EmailAlreadyUsed:
        pass
    fresh = UserService(UserRepository())
    app.dependency_overrides[get_user_service] = lambda: fresh
    c = TestClient(app)
    assert c.post("/users", json={"name": "H", "email": "h@x.com"}).status_code == 201
    assert c.post("/users", json={"name": "H2", "email": "H@x.com"}).status_code == 409
    assert len(c.get("/users").json()) == 1
    app.dependency_overrides.clear()
    print("All checks passed!")
