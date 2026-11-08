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
