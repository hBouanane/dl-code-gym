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
"""

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ProductORM(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    price: Mapped[float]


Base.metadata.create_all(engine)


def get_session():
    with SessionLocal() as session:
        yield session


class ProductIn(BaseModel):
    name: str
    price: float = Field(gt=0)


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    price: float


app = FastAPI()


@app.post("/products", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(body: ProductIn, session: Session = Depends(get_session)):
    product = ProductORM(**body.model_dump())
    session.add(product)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Product name already exists")
    session.refresh(product)
    return product


@app.get("/products", response_model=list[ProductOut])
def list_products(session: Session = Depends(get_session)):
    return session.scalars(select(ProductORM).order_by(ProductORM.id)).all()


@app.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: int, session: Session = Depends(get_session)):
    product = session.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, session: Session = Depends(get_session)):
    product = session.get(ProductORM, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    session.delete(product)
    session.commit()


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
