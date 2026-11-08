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
