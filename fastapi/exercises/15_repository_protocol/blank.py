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

How to practice:
  1. Write everything in blank.py from memory (imports included).
  2. Stuck? Open solution.py, read ONE line, close it, keep going. Note where you got stuck.
  3. Run `python blank.py`. The self-check at the bottom prints "All checks passed!" when you're right.
  4. Redo it tomorrow and the day after, until it comes out without peeking.
"""

# ============ YOUR CODE BELOW ============





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
