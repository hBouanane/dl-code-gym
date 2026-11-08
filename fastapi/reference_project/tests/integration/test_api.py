import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(params=["sqlite://", "memory"], ids=["sqlalchemy", "in-memory"])
def client(request):
    app = create_app(Settings(database_url=request.param))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_loan_cycle(client):
    r = client.post("/books", json={"title": "Clean Architecture", "author": "Robert C. Martin"})
    assert r.status_code == 201
    book_id = r.json()["id"]

    r = client.post(f"/books/{book_id}/borrow", json={"member": "haitam"})
    assert r.status_code == 200 and r.json()["borrower"] == "haitam" and not r.json()["is_available"]

    r = client.post(f"/books/{book_id}/borrow", json={"member": "sara"})
    assert r.status_code == 409 and r.json()["error"] == "BookAlreadyBorrowed"

    assert client.post(f"/books/{book_id}/return").json()["is_available"]
    assert client.post(f"/books/{book_id}/return").status_code == 409


def test_not_found_and_validation(client):
    assert client.get("/books/999").status_code == 404
    assert client.post("/books", json={"title": "", "author": "x"}).status_code == 422
    assert client.post("/books", json={"title": "   ", "author": "x"}).status_code == 422


def test_filter_available(client):
    a = client.post("/books", json={"title": "A", "author": "x"}).json()["id"]
    client.post("/books", json={"title": "B", "author": "y"})
    client.post(f"/books/{a}/borrow", json={"member": "m"})
    titles = [b["title"] for b in client.get("/books", params={"available_only": True}).json()]
    assert titles == ["B"]
