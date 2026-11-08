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
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

app = FastAPI()
BOOKS = {1: "Dune", 2: "Clean Code"}


class OutOfStockError(Exception):
    def __init__(self, book_id: int):
        self.book_id = book_id


@app.exception_handler(OutOfStockError)
def out_of_stock_handler(request: Request, exc: OutOfStockError):
    return JSONResponse(status_code=status.HTTP_409_CONFLICT,
                        content={"error": "out_of_stock", "book_id": exc.book_id})


@app.get("/books/{book_id}")
def get_book(book_id: int):
    if book_id not in BOOKS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return {"id": book_id, "title": BOOKS[book_id]}


@app.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_book(book_id: int):
    if book_id not in BOOKS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    del BOOKS[book_id]


@app.post("/books/{book_id}/buy")
def buy(book_id: int):
    if book_id == 2:
        raise OutOfStockError(book_id)
    return {"bought": book_id}


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
