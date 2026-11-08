from datetime import datetime
from typing import Protocol

from app.domain.models import Book


class BookRepository(Protocol):
    """Outbound port: what the application needs from persistence."""

    def add(self, book: Book) -> Book: ...
    def get(self, book_id: int) -> Book | None: ...
    def update(self, book: Book) -> None: ...
    def list(self, available_only: bool = False) -> list[Book]: ...


class Clock(Protocol):
    """Outbound port for time, so tests can freeze it."""

    def now(self) -> datetime: ...
