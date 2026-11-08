from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import BookAlreadyBorrowed, BookNotBorrowed, BookNotFound, DomainError, InvalidBook

STATUS_BY_ERROR = {
    BookNotFound: status.HTTP_404_NOT_FOUND,
    BookAlreadyBorrowed: status.HTTP_409_CONFLICT,
    BookNotBorrowed: status.HTTP_409_CONFLICT,
    InvalidBook: 422,   # constant name differs across Starlette versions, so use the number
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    def handle_domain_error(request: Request, exc: DomainError):
        code = STATUS_BY_ERROR.get(type(exc), status.HTTP_400_BAD_REQUEST)
        return JSONResponse(status_code=code, content={"error": type(exc).__name__, "detail": str(exc)})
