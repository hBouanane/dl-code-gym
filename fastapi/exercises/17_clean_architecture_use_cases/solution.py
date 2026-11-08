"""
Exercise 17 - Clean architecture: entities, use cases, DTOs, gateways, presenters  [architecture]

Uncle Bob's circles: Entities <- Use cases <- Interface adapters <- Frameworks. Dependencies point INWARD.
1. ENTITIES: @dataclass OrderLine(sku, qty, unit_price) ; @dataclass Order(id, customer_id, lines)
     total property ; rule: an order must have at least one line and qty > 0 (EmptyOrderError / ValueError)
2. USE CASE LAYER:
     @dataclass(frozen=True) PlaceOrderInput(customer_id: str, lines: list[tuple[str, int]])
     @dataclass(frozen=True) PlaceOrderOutput(order_id: str, total: float)
     Gateways (ABC with @abstractmethod): OrderGateway.save(order) ; PriceGateway.price_of(sku) -> float
     class PlaceOrder: __init__(orders: OrderGateway, prices: PriceGateway, id_gen: Callable[[], str])
                       execute(data: PlaceOrderInput) -> PlaceOrderOutput
       (looks up prices, builds the entity, saves it, returns the OUTPUT DTO, never the entity)
3. INTERFACE ADAPTERS: InMemoryOrderGateway ; DictPriceGateway(prices: dict) ;
     present(output) -> dict for the JSON response (presenter: formats total with 2 decimals as string)
4. FRAMEWORK: FastAPI route POST /orders: request model -> PlaceOrderInput -> use case -> presenter
     EmptyOrderError -> 422
Test the use case with fakes and a fixed id_gen (lambda: "order-1"): no web, no DB.
Self-check expects these names: Order, OrderLine, EmptyOrderError, PlaceOrder, PlaceOrderInput, PlaceOrderOutput, InMemoryOrderGateway, DictPriceGateway, present, build_app(use_case) -> FastAPI
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ===== Entities (enterprise business rules) =====
class EmptyOrderError(Exception):
    pass


@dataclass
class OrderLine:
    sku: str
    qty: int
    unit_price: float

    def __post_init__(self):
        if self.qty <= 0:
            raise ValueError("qty must be > 0")


@dataclass
class Order:
    id: str
    customer_id: str
    lines: list[OrderLine] = field(default_factory=list)

    def __post_init__(self):
        if not self.lines:
            raise EmptyOrderError("an order needs at least one line")

    @property
    def total(self) -> float:
        return sum(l.qty * l.unit_price for l in self.lines)


# ===== Use cases (application business rules) =====
@dataclass(frozen=True)
class PlaceOrderInput:
    customer_id: str
    lines: list[tuple[str, int]]


@dataclass(frozen=True)
class PlaceOrderOutput:
    order_id: str
    total: float


class OrderGateway(ABC):
    @abstractmethod
    def save(self, order: Order) -> None: ...


class PriceGateway(ABC):
    @abstractmethod
    def price_of(self, sku: str) -> float: ...


class PlaceOrder:
    def __init__(self, orders: OrderGateway, prices: PriceGateway, id_gen: Callable[[], str]):
        self.orders, self.prices, self.id_gen = orders, prices, id_gen

    def execute(self, data: PlaceOrderInput) -> PlaceOrderOutput:
        lines = [OrderLine(sku, qty, self.prices.price_of(sku)) for sku, qty in data.lines]
        order = Order(id=self.id_gen(), customer_id=data.customer_id, lines=lines)
        self.orders.save(order)
        return PlaceOrderOutput(order_id=order.id, total=order.total)


# ===== Interface adapters =====
class InMemoryOrderGateway(OrderGateway):
    def __init__(self):
        self.saved: dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self.saved[order.id] = order


class DictPriceGateway(PriceGateway):
    def __init__(self, prices: dict[str, float]):
        self.prices = prices

    def price_of(self, sku: str) -> float:
        return self.prices[sku]


def present(output: PlaceOrderOutput) -> dict:
    return {"order_id": output.order_id, "total": f"{output.total:.2f}"}


# ===== Frameworks & drivers =====
class LineIn(BaseModel):
    sku: str
    qty: int


class OrderIn(BaseModel):
    customer_id: str
    lines: list[LineIn]


def build_app(use_case: PlaceOrder) -> FastAPI:
    app = FastAPI()

    @app.post("/orders", status_code=201)
    def place_order(body: OrderIn):
        data = PlaceOrderInput(body.customer_id, [(l.sku, l.qty) for l in body.lines])
        try:
            return present(use_case.execute(data))
        except (EmptyOrderError, ValueError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app


app = build_app(PlaceOrder(InMemoryOrderGateway(), DictPriceGateway({"pen": 1.5, "book": 10.0}),
                           lambda: uuid.uuid4().hex[:8]))


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    gw = InMemoryOrderGateway()
    uc = PlaceOrder(gw, DictPriceGateway({"pen": 1.5, "book": 10.0}), lambda: "order-1")
    out = uc.execute(PlaceOrderInput("c1", [("pen", 2), ("book", 1)]))
    assert out == PlaceOrderOutput("order-1", 13.0) and "order-1" in gw.saved
    assert present(out) == {"order_id": "order-1", "total": "13.00"}
    try:
        uc.execute(PlaceOrderInput("c1", []))
        raise AssertionError("should raise")
    except EmptyOrderError:
        pass
    c = TestClient(build_app(uc))
    assert c.post("/orders", json={"customer_id": "c1", "lines": [{"sku": "pen", "qty": 4}]}).json() == {"order_id": "order-1", "total": "6.00"}
    assert c.post("/orders", json={"customer_id": "c1", "lines": []}).status_code == 422
    assert c.post("/orders", json={"customer_id": "c1", "lines": [{"sku": "pen", "qty": 0}]}).status_code == 422
    print("All checks passed!")
