"""
Exercise 16 - Hexagonal architecture (ports & adapters) end to end  [architecture]

A bank account service. Write these 5 sections IN THIS ORDER (that's the dependency direction):
1. DOMAIN (pure Python, imports nothing from FastAPI/SQL):
     class InsufficientFunds(Exception) ; class AccountNotFound(Exception)
     @dataclass class Account: id: str ; owner: str ; balance: float = 0
        deposit(amount) -> ValueError if amount <= 0
        withdraw(amount) -> ValueError if amount <= 0 ; InsufficientFunds if amount > balance
2. OUTBOUND PORTS (what the core NEEDS): AccountRepository(Protocol): get(id) -> Account | None ; save(account)
                                         Notifier(Protocol): notify(owner: str, message: str)
3. APPLICATION / INBOUND PORT (what the core OFFERS): class AccountService(repo, notifier):
       open_account(owner) -> Account (id = uuid4().hex[:8])
       deposit(id, amount) -> Account ; withdraw(id, amount) -> Account (+ notify when balance < 100)
       raises AccountNotFound for unknown ids
4. OUTBOUND ADAPTERS: InMemoryAccountRepository ; RecordingNotifier (stores messages in a list)
5. INBOUND ADAPTER (HTTP): create_app(service) -> FastAPI app with
       POST /accounts {"owner"} -> 201 ; POST /accounts/{id}/deposit {"amount"} ;
       POST /accounts/{id}/withdraw {"amount"} ; exception handlers:
       AccountNotFound -> 404, InsufficientFunds -> 409, ValueError -> 422
   COMPOSITION ROOT: the only place that wires concrete classes together:
       service = AccountService(InMemoryAccountRepository(), RecordingNotifier()); app = create_app(service)
Self-check expects these names: Account, InsufficientFunds, AccountNotFound, AccountService, InMemoryAccountRepository, RecordingNotifier, create_app
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ================= 1. DOMAIN =================
class InsufficientFunds(Exception):
    pass


class AccountNotFound(Exception):
    pass


@dataclass
class Account:
    id: str
    owner: str
    balance: float = 0

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise InsufficientFunds(f"balance {self.balance} < {amount}")
        self.balance -= amount


# ================= 2. OUTBOUND PORTS =================
class AccountRepository(Protocol):
    def get(self, account_id: str) -> Account | None: ...
    def save(self, account: Account) -> None: ...


class Notifier(Protocol):
    def notify(self, owner: str, message: str) -> None: ...


# ================= 3. APPLICATION (inbound port) =================
class AccountService:
    LOW_BALANCE = 100

    def __init__(self, repo: AccountRepository, notifier: Notifier):
        self.repo = repo
        self.notifier = notifier

    def _get(self, account_id: str) -> Account:
        account = self.repo.get(account_id)
        if account is None:
            raise AccountNotFound(account_id)
        return account

    def open_account(self, owner: str) -> Account:
        account = Account(id=uuid.uuid4().hex[:8], owner=owner)
        self.repo.save(account)
        return account

    def deposit(self, account_id: str, amount: float) -> Account:
        account = self._get(account_id)
        account.deposit(amount)
        self.repo.save(account)
        return account

    def withdraw(self, account_id: str, amount: float) -> Account:
        account = self._get(account_id)
        account.withdraw(amount)
        self.repo.save(account)
        if account.balance < self.LOW_BALANCE:
            self.notifier.notify(account.owner, f"Low balance: {account.balance}")
        return account


# ================= 4. OUTBOUND ADAPTERS =================
class InMemoryAccountRepository:
    def __init__(self):
        self._data: dict[str, Account] = {}

    def get(self, account_id: str) -> Account | None:
        return self._data.get(account_id)

    def save(self, account: Account) -> None:
        self._data[account.id] = account


class RecordingNotifier:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def notify(self, owner: str, message: str) -> None:
        self.messages.append((owner, message))


# ================= 5. INBOUND ADAPTER (HTTP) =================
class OpenAccountIn(BaseModel):
    owner: str


class AmountIn(BaseModel):
    amount: float


def create_app(service: AccountService) -> FastAPI:
    app = FastAPI(title="Bank (hexagonal)")

    @app.exception_handler(AccountNotFound)
    def not_found(request: Request, exc: AccountNotFound):
        return JSONResponse(status_code=404, content={"detail": "account not found"})

    @app.exception_handler(InsufficientFunds)
    def insufficient(request: Request, exc: InsufficientFunds):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def invalid(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/accounts", status_code=status.HTTP_201_CREATED)
    def open_account(body: OpenAccountIn):
        return service.open_account(body.owner)

    @app.post("/accounts/{account_id}/deposit")
    def deposit(account_id: str, body: AmountIn):
        return service.deposit(account_id, body.amount)

    @app.post("/accounts/{account_id}/withdraw")
    def withdraw(account_id: str, body: AmountIn):
        return service.withdraw(account_id, body.amount)

    return app


# ================= COMPOSITION ROOT =================
notifier = RecordingNotifier()
service = AccountService(InMemoryAccountRepository(), notifier)
app = create_app(service)


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    acc = Account("x", "me", 50)
    try:
        acc.withdraw(80)
        raise AssertionError("should raise")
    except InsufficientFunds:
        pass
    n = RecordingNotifier()
    svc = AccountService(InMemoryAccountRepository(), n)
    a = svc.open_account("haitam")
    svc.deposit(a.id, 150)
    assert svc.withdraw(a.id, 100).balance == 50 and n.messages == [("haitam", "Low balance: 50")]
    try:
        svc.deposit("nope", 10)
        raise AssertionError("should raise")
    except AccountNotFound:
        pass
    c = TestClient(create_app(AccountService(InMemoryAccountRepository(), RecordingNotifier())))
    aid = c.post("/accounts", json={"owner": "sara"}).json()["id"]
    assert c.post(f"/accounts/{aid}/deposit", json={"amount": 200}).json()["balance"] == 200
    assert c.post(f"/accounts/{aid}/withdraw", json={"amount": 500}).status_code == 409
    assert c.post(f"/accounts/{aid}/deposit", json={"amount": -5}).status_code == 422
    assert c.post("/accounts/unknown/deposit", json={"amount": 5}).status_code == 404
    print("All checks passed!")
