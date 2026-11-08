"""
Exercise 07 - SQLAlchemy 2.0 ORM + session dependency + CRUD  [database]

1. engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
   (in-memory DB shared across threads; a real app uses "sqlite:///app.db" or a Postgres URL)
   SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
2. class Base(DeclarativeBase) ; class ProductORM(Base): __tablename__ = "products"
     id: Mapped[int] = mapped_column(primary_key=True)
     name: Mapped[str] = mapped_column(String(100), unique=True)
     price: Mapped[float]
   Base.metadata.create_all(engine)
3. get_session(): with SessionLocal() as session: yield session
4. Pydantic: ProductIn(name, price>0) ; ProductOut(id, name, price) with
   model_config = ConfigDict(from_attributes=True)   (read from ORM objects)
5. Endpoints: POST /products (201; 409 if the name already exists -> catch IntegrityError, rollback)
              GET /products (select(ProductORM).order_by(ProductORM.id)) ; GET /products/{id} (session.get -> 404)
              DELETE /products/{id}
Remember the 2.0 style: session.scalars(select(...)).all(), session.add, session.commit, session.refresh
Self-check expects these names: app, engine, Base

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
    c = TestClient(app)
    r = c.post("/products", json={"name": "keyboard", "price": 49.9})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert c.post("/products", json={"name": "keyboard", "price": 10}).status_code == 409
    assert c.post("/products", json={"name": "mouse", "price": 0}).status_code == 422
    assert c.get(f"/products/{pid}").json()["name"] == "keyboard"
    assert any(p["name"] == "keyboard" for p in c.get("/products").json())
    assert c.delete(f"/products/{pid}").status_code == 204
    assert c.get(f"/products/{pid}").status_code == 404
    print("All checks passed!")
