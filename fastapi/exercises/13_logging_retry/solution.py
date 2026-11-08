"""
Exercise 13 - Production basics: logging and retries (tenacity)  [production]

1. Logging: logger = logging.getLogger("app"); configure ONCE at startup with
   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
   Never use print() in a service. Log with context: logger.info("order created", extra={"order_id": 1})
   Log exceptions with logger.exception("...") inside except (it adds the traceback).
2. Retries with tenacity for a flaky dependency:
     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.01, max=0.1),
            retry=retry_if_exception_type(ConnectionError), reraise=True)
     def call_payment_service(): ...  (fails twice with ConnectionError, then returns "paid")
   CALLS counts the attempts -> 3.
3. A function that always fails -> after 3 attempts the ConnectionError is re-raised.
4. Never retry on errors that won't fix themselves (ValueError, 4xx) -> a ValueError is raised
   immediately, after 1 attempt.
Self-check expects these names: call_payment_service, always_fails, bad_input, CALLS (dict of counters)
"""

import logging

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("app")
CALLS = {"payment": 0, "always": 0, "bad": 0}

retry_network = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.01, max=0.1),
    retry=retry_if_exception_type(ConnectionError),
    reraise=True,
)


@retry_network
def call_payment_service() -> str:
    CALLS["payment"] += 1
    if CALLS["payment"] < 3:
        logger.warning("payment service unavailable, attempt %d", CALLS["payment"])
        raise ConnectionError("payment service down")
    return "paid"


@retry_network
def always_fails():
    CALLS["always"] += 1
    raise ConnectionError("down forever")


@retry_network
def bad_input():
    CALLS["bad"] += 1
    raise ValueError("invalid card number")


print(call_payment_service())
try:
    always_fails()
except ConnectionError:
    logger.exception("gave up after %d attempts", CALLS["always"])


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    for k in CALLS:
        CALLS[k] = 0
    assert call_payment_service() == "paid" and CALLS["payment"] == 3
    try:
        always_fails()
        raise AssertionError("should raise")
    except ConnectionError:
        assert CALLS["always"] == 3
    try:
        bad_input()
    except ValueError:
        assert CALLS["bad"] == 1
    print("All checks passed!")
