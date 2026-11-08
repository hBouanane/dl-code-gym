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
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


# ---------- events ----------
@dataclass(frozen=True)
class UserRegistered:
    user_id: str
    email: str


@dataclass(frozen=True)
class UserDeactivated:
    user_id: str


# ---------- aggregate ----------
@dataclass
class User:
    id: str
    email: str
    active: bool = True
    events: list = field(default_factory=list)

    @classmethod
    def register(cls, user_id: str, email: str) -> "User":
        user = cls(user_id, email)
        user.events.append(UserRegistered(user_id, email))
        return user

    def deactivate(self) -> None:
        if not self.active:
            raise ValueError("already inactive")
        self.active = False
        self.events.append(UserDeactivated(self.id))


# ---------- bus ----------
class MessageBus:
    def __init__(self, handlers: dict[type, list[Callable]]):
        self.handlers = defaultdict(list, handlers)

    def publish(self, event) -> None:
        for handler in self.handlers[type(event)]:
            handler(event)

    def publish_all(self, aggregate) -> None:
        while aggregate.events:
            self.publish(aggregate.events.pop(0))


# ---------- handlers (side effects live at the edges) ----------
OUTBOX: list[str] = []
ANALYTICS: list[str] = []
AUDIT: list[str] = []


def send_welcome_email(event: UserRegistered):
    OUTBOX.append(f"Welcome {event.email}")


def track_signup(event: UserRegistered):
    ANALYTICS.append(f"signup:{event.user_id}")


def log_deactivation(event: UserDeactivated):
    AUDIT.append(f"deactivated:{event.user_id}")


HANDLERS = {
    UserRegistered: [send_welcome_email, track_signup],
    UserDeactivated: [log_deactivation],
}


# ---------- service ----------
def register_user(bus: MessageBus, user_id: str, email: str) -> User:
    user = User.register(user_id, email)
    # (save the user with a repository / unit of work here)
    bus.publish_all(user)
    return user


bus = MessageBus(HANDLERS)
u = register_user(bus, "u1", "h@x.com")
u.deactivate()
bus.publish_all(u)
print(OUTBOX, ANALYTICS, AUDIT)


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
