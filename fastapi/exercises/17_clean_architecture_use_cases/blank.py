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
