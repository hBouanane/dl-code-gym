from datetime import datetime, timezone

import pytest

from app.adapters.outbound.memory import InMemoryBookRepository
from app.application.services import LibraryService

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FrozenClock:
    def now(self) -> datetime:
        return FIXED_NOW


@pytest.fixture
def service() -> LibraryService:
    return LibraryService(books=InMemoryBookRepository(), clock=FrozenClock())
