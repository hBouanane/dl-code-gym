from app.application.ports import BookRepository, Clock
from app.domain.errors import BookNotFound
from app.domain.models import Book


class LibraryService:
    """Inbound port: the use cases the outside world can trigger."""

    def __init__(self, books: BookRepository, clock: Clock):
        self.books = books
        self.clock = clock

    def add_book(self, title: str, author: str) -> Book:
        return self.books.add(Book(title=title, author=author))

    def get_book(self, book_id: int) -> Book:
        book = self.books.get(book_id)
        if book is None:
            raise BookNotFound(book_id)
        return book

    def list_books(self, available_only: bool = False) -> list[Book]:
        return self.books.list(available_only=available_only)

    def borrow(self, book_id: int, member: str) -> Book:
        book = self.get_book(book_id)
        book.borrow(member, self.clock.now())
        self.books.update(book)
        return book

    def give_back(self, book_id: int) -> Book:
        book = self.get_book(book_id)
        book.give_back()
        self.books.update(book)
        return book
