class DomainError(Exception):
    """Base class for every business-rule violation."""


class BookNotFound(DomainError):
    def __init__(self, book_id: int):
        super().__init__(f"Book {book_id} not found")
        self.book_id = book_id


class BookAlreadyBorrowed(DomainError):
    def __init__(self, book_id: int):
        super().__init__(f"Book {book_id} is already borrowed")
        self.book_id = book_id


class BookNotBorrowed(DomainError):
    def __init__(self, book_id: int):
        super().__init__(f"Book {book_id} is not borrowed")
        self.book_id = book_id


class InvalidBook(DomainError):
    pass
