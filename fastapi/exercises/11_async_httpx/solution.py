"""
Exercise 11 - async endpoints + httpx.AsyncClient + asyncio.gather (+ mocking)  [async]

Calling other services without blocking.
1. async def fetch_price(client: httpx.AsyncClient, symbol: str) -> float:
       r = await client.get(f"https://prices.example.com/{symbol}"); r.raise_for_status()
       return r.json()["price"]
2. async def fetch_all(client, symbols) -> dict: run all requests CONCURRENTLY with
   asyncio.gather(*[fetch_price(client, s) for s in symbols])
3. Dependency get_http_client() -> yields an httpx.AsyncClient (async with ...)
   GET /portfolio?symbols=AAPL&symbols=MSFT -> {"prices": {...}, "total": ...}
   if the upstream fails (httpx.HTTPStatusError) -> 502 Bad Gateway
4. Testing without internet: httpx.MockTransport(handler) where handler(request) returns
   httpx.Response(200, json={"price": ...}) (404 for unknown symbols).
   Override get_http_client in the app to use the mocked client.
Rule: in `async def` endpoints never call blocking code (requests, time.sleep) -> use httpx / asyncio.sleep.
Self-check expects these names: app, fetch_all, get_http_client, mock_handler
"""

import asyncio

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query

PRICES = {"AAPL": 190.0, "MSFT": 410.0, "NVDA": 120.0}


def mock_handler(request: httpx.Request) -> httpx.Response:
    symbol = request.url.path.strip("/")
    if symbol not in PRICES:
        return httpx.Response(404, json={"error": "unknown symbol"})
    return httpx.Response(200, json={"price": PRICES[symbol]})


async def fetch_price(client: httpx.AsyncClient, symbol: str) -> float:
    r = await client.get(f"https://prices.example.com/{symbol}")
    r.raise_for_status()
    return r.json()["price"]


async def fetch_all(client: httpx.AsyncClient, symbols: list[str]) -> dict[str, float]:
    prices = await asyncio.gather(*[fetch_price(client, s) for s in symbols])
    return dict(zip(symbols, prices))


async def get_http_client():
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


app = FastAPI()


@app.get("/portfolio")
async def portfolio(symbols: list[str] = Query(), client: httpx.AsyncClient = Depends(get_http_client)):
    try:
        prices = await fetch_all(client, symbols)
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="Price service error")
    return {"prices": prices, "total": sum(prices.values())}


async def _demo():
    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        print(await fetch_all(client, ["AAPL", "MSFT"]))

asyncio.run(_demo())


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    import asyncio
    import httpx


    async def _mocked_client():
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
            yield client


    async def _t():
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
            return await fetch_all(client, ["AAPL", "NVDA"])

    assert asyncio.run(_t()) == {"AAPL": 190.0, "NVDA": 120.0}
    app.dependency_overrides[get_http_client] = _mocked_client
    c = TestClient(app)
    assert c.get("/portfolio?symbols=AAPL&symbols=MSFT").json() == {"prices": {"AAPL": 190.0, "MSFT": 410.0}, "total": 600.0}
    assert c.get("/portfolio?symbols=XXX").status_code == 502
    app.dependency_overrides.clear()
    print("All checks passed!")
