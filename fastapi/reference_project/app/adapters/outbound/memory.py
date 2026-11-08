import copy

from app.domain.models import Book


class InMemoryBookRepository:
    def __init__(self):
        self._books: dict[int, Book] = {}
        self._next_id = 1

    def add(self, book: Book) -> Book:
        book.id = self._next_id
        self._next_id += 1
        self._books[book.id] = copy.deepcopy(book)
        return book

    def get(self, book_id: int) -> Book | None:
        book = self._books.get(book_id)
        return copy.deepcopy(book) if book else None

    def update(self, book: Book) -> None:
        self._books[book.id] = copy.deepcopy(book)

    def list(self, available_only: bool = False) -> list[Book]:
        books = [copy.deepcopy(b) for b in self._books.values()]
        return [b for b in books if b.is_available] if available_only else books
