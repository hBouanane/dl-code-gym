"""
Exercise 18 - Unit of Work: atomic transactions behind an interface  [architecture]

Problem: a money transfer touches 2 accounts. Either both are saved or none.
1. Port: class AbstractUnitOfWork(ABC):
       accounts: an object with get(id) / add(account)
       __enter__ returns self ; __exit__(*args) calls self.rollback() (rollback whatever was not committed)
       abstract commit() and rollback()
2. InMemoryUnitOfWork: keeps `committed` data (dict of id -> balance); works on a COPY inside the
   transaction; commit() copies back; rollback() throws the copy away. (`committed_count` counter)
3. SqlAlchemyUnitOfWork(session_factory): __enter__ opens a session and builds a repository on it,
   commit() -> session.commit(), rollback() -> session.rollback(), __exit__ also closes the session.
4. Service: transfer(uow, from_id, to_id, amount):
       with uow:
           src, dst = uow.accounts.get(from_id), uow.accounts.get(to_id)
           src.balance -= amount ; if src.balance < 0: raise ValueError("insufficient funds")
           dst.balance += amount
           uow.commit()
   If anything raises before commit -> automatic rollback, balances unchanged.
Use a plain @dataclass Account(id: str, balance: float) and, for SQLAlchemy, a mapped AccountRow.
Self-check expects these names: transfer, InMemoryUnitOfWork(initial: dict[str, float]), make_sqlalchemy_uow(initial: dict[str, float]), balances(uow) -> dict
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


@dataclass
class Account:
    id: str
    balance: float


class AbstractUnitOfWork(ABC):
    accounts: object

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback()

    @abstractmethod
    def commit(self): ...

    @abstractmethod
    def rollback(self): ...


# ---------- in-memory ----------
class _DictRepo:
    def __init__(self, data: dict[str, Account]):
        self.data = data

    def get(self, account_id: str) -> Account:
        return self.data[account_id]

    def add(self, account: Account) -> None:
        self.data[account.id] = account


class InMemoryUnitOfWork(AbstractUnitOfWork):
    def __init__(self, initial: dict[str, float]):
        self.committed = {k: Account(k, v) for k, v in initial.items()}
        self.committed_count = 0

    def __enter__(self):
        self.accounts = _DictRepo(copy.deepcopy(self.committed))
        return self

    def commit(self):
        self.committed = copy.deepcopy(self.accounts.data)
        self.committed_count += 1

    def rollback(self):
        pass   # the working copy is simply discarded


# ---------- SQLAlchemy ----------
class Base(DeclarativeBase):
    pass


class AccountRow(Base):
    __tablename__ = "accounts"
    id: Mapped[str] = mapped_column(primary_key=True)
    balance: Mapped[float]


class _SqlRepo:
    def __init__(self, session: Session):
        self.session = session

    def get(self, account_id: str) -> AccountRow:
        return self.session.get(AccountRow, account_id)

    def add(self, account: Account) -> None:
        self.session.add(AccountRow(id=account.id, balance=account.balance))


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()
        self.accounts = _SqlRepo(self.session)
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()


def make_sqlalchemy_uow(initial: dict[str, float]) -> SqlAlchemyUnitOfWork:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        s.add_all([AccountRow(id=k, balance=v) for k, v in initial.items()])
        s.commit()
    return SqlAlchemyUnitOfWork(factory)


# ---------- service ----------
def transfer(uow: AbstractUnitOfWork, from_id: str, to_id: str, amount: float) -> None:
    with uow:
        src, dst = uow.accounts.get(from_id), uow.accounts.get(to_id)
        src.balance -= amount
        if src.balance < 0:
            raise ValueError("insufficient funds")
        dst.balance += amount
        uow.commit()


def balances(uow: AbstractUnitOfWork) -> dict[str, float]:
    if isinstance(uow, InMemoryUnitOfWork):
        return {k: a.balance for k, a in uow.committed.items()}
    with uow.session_factory() as s:
        return {r.id: r.balance for r in s.scalars(select(AccountRow)).all()}


for uow in [InMemoryUnitOfWork({"a": 100, "b": 0}), make_sqlalchemy_uow({"a": 100, "b": 0})]:
    transfer(uow, "a", "b", 30)
    try:
        transfer(uow, "a", "b", 500)
    except ValueError:
        pass
    print(type(uow).__name__, balances(uow))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    for u in [InMemoryUnitOfWork({"a": 100, "b": 0}), make_sqlalchemy_uow({"a": 100, "b": 0})]:
        transfer(u, "a", "b", 40)
        assert balances(u) == {"a": 60, "b": 40}
        try:
            transfer(u, "a", "b", 1000)
            raise AssertionError("should raise")
        except ValueError:
            pass
        assert balances(u) == {"a": 60, "b": 40}, "failed transfer must roll back"
    print("All checks passed!")
