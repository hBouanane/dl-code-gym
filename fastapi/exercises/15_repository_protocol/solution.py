"""
Exercise 15 - Repository pattern with typing.Protocol: one port, two adapters  [architecture]

1. Domain object: @dataclass class Product: name: str ; price: float ; id: int | None = None
2. PORT (an interface): class ProductRepository(Protocol):
       def add(self, product: Product) -> Product: ...
       def get(self, product_id: int) -> Product | None: ...
       def list(self) -> list[Product]: ...
3. ADAPTER 1: InMemoryProductRepository (dict)
   ADAPTER 2: SqlAlchemyProductRepository(session) with its OWN ORM model ProductRow and
              mapping functions _to_domain(row) / _to_row(product)
              (the domain dataclass never imports SQLAlchemy!)
4. A service function that depends ONLY on the port:
       def cheapest(repo: ProductRepository) -> Product | None
5. contract_test(repo) runs the same assertions against any adapter -> run it on both.
Why: you can switch Postgres <-> in-memory <-> an external API without touching business code.
Self-check expects these names: Product, InMemoryProductRepository, SqlAlchemyProductRepository, make_sqlalchemy_repo() -> repo, cheapest
"""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# ---------- domain ----------
@dataclass
class Product:
    name: str
    price: float
    id: int | None = None


# ---------- port ----------
class ProductRepository(Protocol):
    def add(self, product: Product) -> Product: ...
    def get(self, product_id: int) -> Product | None: ...
    def list(self) -> list[Product]: ...


# ---------- adapter 1: in memory ----------
class InMemoryProductRepository:
    def __init__(self):
        self._items: dict[int, Product] = {}

    def add(self, product: Product) -> Product:
        product.id = len(self._items) + 1
        self._items[product.id] = product
        return product

    def get(self, product_id: int) -> Product | None:
        return self._items.get(product_id)

    def list(self) -> list[Product]:
        return list(self._items.values())


# ---------- adapter 2: SQLAlchemy ----------
class Base(DeclarativeBase):
    pass


class ProductRow(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    price: Mapped[float]


def _to_domain(row: ProductRow) -> Product:
    return Product(id=row.id, name=row.name, price=row.price)


class SqlAlchemyProductRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, product: Product) -> Product:
        row = ProductRow(name=product.name, price=product.price)
        self.session.add(row)
        self.session.commit()
        return _to_domain(row)

    def get(self, product_id: int) -> Product | None:
        row = self.session.get(ProductRow, product_id)
        return _to_domain(row) if row else None

    def list(self) -> list[Product]:
        return [_to_domain(r) for r in self.session.scalars(select(ProductRow)).all()]


def make_sqlalchemy_repo() -> SqlAlchemyProductRepository:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return SqlAlchemyProductRepository(Session(engine))


# ---------- business logic: depends on the port only ----------
def cheapest(repo: ProductRepository) -> Product | None:
    return min(repo.list(), key=lambda p: p.price, default=None)


def contract_test(repo: ProductRepository):
    a = repo.add(Product("pen", 2.5))
    repo.add(Product("book", 12.0))
    assert a.id is not None
    assert repo.get(a.id).name == "pen"
    assert repo.get(999) is None
    assert len(repo.list()) == 2
    assert cheapest(repo).name == "pen"


for repo in [InMemoryProductRepository(), make_sqlalchemy_repo()]:
    contract_test(repo)
    print(type(repo).__name__, "passes the contract")


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    for r in [InMemoryProductRepository(), make_sqlalchemy_repo()]:
        assert cheapest(r) is None
        p = r.add(Product("pen", 2.5))
        r.add(Product("book", 12.0))
        assert r.get(p.id) == Product("pen", 2.5, p.id)
        assert r.get(999) is None
        assert [x.name for x in r.list()] == ["pen", "book"]
        assert cheapest(r).name == "pen"
        assert isinstance(r.list()[0], Product)
    print("All checks passed!")
