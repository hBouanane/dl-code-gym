from dataclasses import dataclass
from datetime import datetime

from app.domain.errors import BookAlreadyBorrowed, BookNotBorrowed, InvalidBook


@dataclass
class Book:
    """Entity: identity + business rules. Plain Python, no framework."""

    title: str
    author: str
    id: int | None = None
    borrower: str | None = None
    borrowed_at: datetime | None = None

    def __post_init__(self):
        if not self.title.strip() or not self.author.strip():
            raise InvalidBook("title and author are required")

    @property
    def is_available(self) -> bool:
        return self.borrower is None

    def borrow(self, member: str, now: datetime) -> None:
        if not self.is_available:
            raise BookAlreadyBorrowed(self.id)
        self.borrower = member
        self.borrowed_at = now

    def give_back(self) -> None:
        if self.is_available:
            raise BookNotBorrowed(self.id)
        self.borrower = None
        self.borrowed_at = None
