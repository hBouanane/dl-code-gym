"""
Exercise 03 - Status codes, HTTPException, custom exception handlers  [basics]

1. BOOKS = {1: "Dune", 2: "Clean Code"}
   GET /books/{book_id}: 404 with HTTPException(status_code=404, detail="Book not found") if missing
2. DELETE /books/{book_id}: status_code=204 (No Content), returns nothing; 404 if missing
3. Custom domain exception: class OutOfStockError(Exception) with a `book_id` attribute.
   POST /books/{book_id}/buy raises OutOfStockError when book_id == 2.
   Register @app.exception_handler(OutOfStockError) that returns
   JSONResponse(status_code=409, content={"error": "out_of_stock", "book_id": ...})
4. Use `status.HTTP_404_NOT_FOUND` style constants instead of magic numbers.
Why this matters: business code raises domain errors; ONE place maps them to HTTP codes.
Self-check expects these names: app

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
    c = TestClient(app)
    assert c.get("/books/1").json()["title"] == "Dune"
    r = c.get("/books/99")
    assert r.status_code == 404 and r.json() == {"detail": "Book not found"}
    assert c.post("/books/2/buy").status_code == 409
    assert c.post("/books/2/buy").json() == {"error": "out_of_stock", "book_id": 2}
    assert c.post("/books/1/buy").json() == {"bought": 1}
    assert c.delete("/books/1").status_code == 204
    assert c.delete("/books/1").status_code == 404
    print("All checks passed!")
