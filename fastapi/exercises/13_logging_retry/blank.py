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
