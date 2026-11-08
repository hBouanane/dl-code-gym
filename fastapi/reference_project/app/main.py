from fastapi import FastAPI

from app.adapters.inbound.http.errors import register_error_handlers
from app.adapters.inbound.http.routes import router
from app.adapters.outbound.clock import SystemClock
from app.adapters.outbound.memory import InMemoryBookRepository
from app.adapters.outbound.sqlalchemy_repo import SqlAlchemyBookRepository, make_session_factory
from app.application.ports import Clock
from app.application.services import LibraryService
from app.config import Settings, get_settings


def build_service(settings: Settings, clock: Clock | None = None) -> LibraryService:
    """Composition root: the ONLY place that knows the concrete adapters."""
    if settings.database_url == "memory":
        repo = InMemoryBookRepository()
    else:
        repo = SqlAlchemyBookRepository(make_session_factory(settings.database_url))
    return LibraryService(books=repo, clock=clock or SystemClock())


def create_app(settings: Settings | None = None, service: LibraryService | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.library_service = service or build_service(settings)
    app.include_router(router)
    register_error_handlers(app)

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    return app


app = create_app()
