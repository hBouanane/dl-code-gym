import pytest

from app.domain.errors import BookAlreadyBorrowed, BookNotBorrowed, BookNotFound, InvalidBook
from app.domain.models import Book
from tests.conftest import FIXED_NOW


def test_book_requires_title():
    with pytest.raises(InvalidBook):
        Book(title="  ", author="x")


def test_add_and_get(service):
    book = service.add_book("Dune", "Herbert")
    assert service.get_book(book.id).title == "Dune"


def test_borrow_sets_borrower_and_time(service):
    book = service.add_book("Dune", "Herbert")
    borrowed = service.borrow(book.id, "haitam")
    assert borrowed.borrower == "haitam" and borrowed.borrowed_at == FIXED_NOW
    assert not service.get_book(book.id).is_available


def test_cannot_borrow_twice(service):
    book = service.add_book("Dune", "Herbert")
    service.borrow(book.id, "a")
    with pytest.raises(BookAlreadyBorrowed):
        service.borrow(book.id, "b")


def test_return_flow(service):
    book = service.add_book("Dune", "Herbert")
    with pytest.raises(BookNotBorrowed):
        service.give_back(book.id)
    service.borrow(book.id, "a")
    assert service.give_back(book.id).is_available


def test_unknown_book(service):
    with pytest.raises(BookNotFound):
        service.borrow(42, "a")


def test_list_available_only(service):
    a = service.add_book("A", "x")
    service.add_book("B", "y")
    service.borrow(a.id, "m")
    assert [b.title for b in service.list_books(available_only=True)] == ["B"]
