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
