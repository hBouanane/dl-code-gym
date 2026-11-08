"""
Exercise 12 - Testing with pytest: fixtures, TestClient, overrides, parametrize  [testing]

This file is a TEST file. Write the app AND its tests in it, then run:  python -m pytest blank.py -q
App (keep it small): GET /square/{n} -> {"result": n*n} ; POST /notes stores into a repository
obtained via Depends(get_repo) where get_repo returns a global dict-based NoteRepo.
Tests to write:
1. @pytest.fixture client(): override get_repo with a FRESH NoteRepo per test,
   yield TestClient(app), then app.dependency_overrides.clear()
2. test_square with @pytest.mark.parametrize("n,expected", [(2, 4), (3, 9), (-4, 16)])
3. test_square_rejects_text -> 422
4. test_create_note_uses_fresh_repo: post a note, assert the repo has exactly 1 note
   (run twice in different tests -> each test is isolated thanks to the fixture)
5. test_note_title_required -> 422
The self-check runs pytest on this file and expects every test to pass (at least 6 tests).
Self-check expects these names: app, get_repo, NoteRepo, and tests named test_*
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class NoteIn(BaseModel):
    title: str


class NoteRepo:
    def __init__(self):
        self.notes: list[str] = []

    def add(self, title: str) -> int:
        self.notes.append(title)
        return len(self.notes)


_global_repo = NoteRepo()


def get_repo() -> NoteRepo:
    return _global_repo


app = FastAPI()


@app.get("/square/{n}")
def square(n: int):
    return {"result": n * n}


@app.post("/notes", status_code=201)
def create_note(body: NoteIn, repo: NoteRepo = Depends(get_repo)):
    return {"id": repo.add(body.title)}


@pytest.fixture
def repo():
    return NoteRepo()


@pytest.fixture
def client(repo):
    app.dependency_overrides[get_repo] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("n,expected", [(2, 4), (3, 9), (-4, 16)])
def test_square(client, n, expected):
    assert client.get(f"/square/{n}").json() == {"result": expected}


def test_square_rejects_text(client):
    assert client.get("/square/abc").status_code == 422


def test_create_note_uses_fresh_repo(client, repo):
    r = client.post("/notes", json={"title": "hello"})
    assert r.status_code == 201 and r.json() == {"id": 1}
    assert repo.notes == ["hello"]


def test_create_note_again_is_isolated(client, repo):
    client.post("/notes", json={"title": "second"})
    assert repo.notes == ["second"]


def test_note_title_required(client):
    assert client.post("/notes", json={}).status_code == 422


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    import sys
    import pytest


    class _Count:
        passed = 0
        failed = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    _Count.passed += 1
                elif report.failed:
                    _Count.failed += 1


    code = pytest.main([__file__, "-q", "-p", "no:cacheprovider"], plugins=[_Count()])
    assert code == 0 and _Count.failed == 0, "some tests failed"
    assert _Count.passed >= 6, f"write at least 6 tests (found {_Count.passed})"
    print("All checks passed!")
