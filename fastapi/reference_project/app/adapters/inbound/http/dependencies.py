from typing import Annotated

from fastapi import Depends, Request

from app.application.services import LibraryService


def get_library_service(request: Request) -> LibraryService:
    # the service is built once by the composition root and stored on app.state
    return request.app.state.library_service


LibraryServiceDep = Annotated[LibraryService, Depends(get_library_service)]
