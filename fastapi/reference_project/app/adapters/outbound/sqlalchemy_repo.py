from datetime import datetime

from sqlalchemy import DateTime, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.models import Book


class Base(DeclarativeBase):
    pass


class BookRow(Base):
    """Persistence model. Lives in the adapter, never leaks into the domain."""

    __tablename__ = "books"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    author: Mapped[str] = mapped_column(String(200))
    borrower: Mapped[str | None] = mapped_column(String(100), default=None)
    borrowed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


def _to_domain(row: BookRow) -> Book:
    return Book(id=row.id, title=row.title, author=row.author, borrower=row.borrower, borrowed_at=row.borrowed_at)


def make_session_factory(database_url: str):
    kwargs = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)   # real projects: Alembic migrations instead
    return sessionmaker(bind=engine, expire_on_commit=False)


class SqlAlchemyBookRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def add(self, book: Book) -> Book:
        with self.session_factory() as s:
            row = BookRow(title=book.title, author=book.author)
            s.add(row)
            s.commit()
            book.id = row.id
        return book

    def get(self, book_id: int) -> Book | None:
        with self.session_factory() as s:
            row = s.get(BookRow, book_id)
            return _to_domain(row) if row else None

    def update(self, book: Book) -> None:
        with self.session_factory() as s:
            row = s.get(BookRow, book.id)
            row.title, row.author = book.title, book.author
            row.borrower, row.borrowed_at = book.borrower, book.borrowed_at
            s.commit()

    def list(self, available_only: bool = False) -> list[Book]:
        stmt = select(BookRow).order_by(BookRow.id)
        if available_only:
            stmt = stmt.where(BookRow.borrower.is_(None))
        with self.session_factory() as s:
            return [_to_domain(r) for r in s.scalars(stmt).all()]
