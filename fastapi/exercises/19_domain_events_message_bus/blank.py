"""
Exercise 19 - Domain events + a tiny message bus (decoupled side effects)  [architecture]

Instead of the service calling email/analytics/etc. directly, the domain RECORDS events and a bus
dispatches them to handlers. Adding a new side effect = adding a handler, no core change.
1. Events: @dataclass(frozen=True) UserRegistered(user_id: str, email: str) ;
           @dataclass(frozen=True) UserDeactivated(user_id: str)
2. Aggregate: @dataclass User(id, email, active=True, events: list = field(default_factory=list))
     @classmethod register(cls, id, email) -> creates the user and appends UserRegistered
     deactivate() -> raises ValueError if already inactive, appends UserDeactivated
3. class MessageBus: __init__(handlers: dict[type, list[Callable]])
     publish(event): call every handler registered for type(event)
     publish_all(aggregate): publish each event in aggregate.events then clear the list
4. Handlers: send_welcome_email(event) -> append to OUTBOX ; track_signup(event) -> append to ANALYTICS ;
             log_deactivation(event) -> append to AUDIT
5. Service function register_user(bus, user_id, email) -> User: creates it, then bus.publish_all(user)
Self-check expects these names: User, UserRegistered, UserDeactivated, MessageBus, register_user, OUTBOX, ANALYTICS, AUDIT, HANDLERS (the handlers dict)

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
    OUTBOX.clear(); ANALYTICS.clear(); AUDIT.clear()
    b = MessageBus(HANDLERS)
    usr = register_user(b, "u9", "a@b.c")
    assert OUTBOX == ["Welcome a@b.c"] and ANALYTICS == ["signup:u9"] and usr.events == []
    usr.deactivate()
    assert usr.events == [UserDeactivated("u9")]
    b.publish_all(usr)
    assert AUDIT == ["deactivated:u9"] and usr.events == []
    try:
        usr.deactivate()
        raise AssertionError("should raise")
    except ValueError:
        pass
    print("All checks passed!")
