from fastapi import APIRouter, status

from app.adapters.inbound.http.dependencies import LibraryServiceDep
from app.adapters.inbound.http.schemas import BookIn, BookOut, BorrowIn

router = APIRouter(prefix="/books", tags=["books"])


@router.post("", response_model=BookOut, status_code=status.HTTP_201_CREATED)
def add_book(body: BookIn, service: LibraryServiceDep):
    return service.add_book(body.title, body.author)


@router.get("", response_model=list[BookOut])
def list_books(service: LibraryServiceDep, available_only: bool = False):
    return service.list_books(available_only=available_only)


@router.get("/{book_id}", response_model=BookOut)
def get_book(book_id: int, service: LibraryServiceDep):
    return service.get_book(book_id)


@router.post("/{book_id}/borrow", response_model=BookOut)
def borrow(book_id: int, body: BorrowIn, service: LibraryServiceDep):
    return service.borrow(book_id, body.member)


@router.post("/{book_id}/return", response_model=BookOut)
def give_back(book_id: int, service: LibraryServiceDep):
    return service.give_back(book_id)
