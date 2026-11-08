"""
Exercise 18 - Unit of Work: atomic transactions behind an interface  [architecture]

Problem: a money transfer touches 2 accounts. Either both are saved or none.
1. Port: class AbstractUnitOfWork(ABC):
       accounts: an object with get(id) / add(account)
       __enter__ returns self ; __exit__(*args) calls self.rollback() (rollback whatever was not committed)
       abstract commit() and rollback()
2. InMemoryUnitOfWork: keeps `committed` data (dict of id -> balance); works on a COPY inside the
   transaction; commit() copies back; rollback() throws the copy away. (`committed_count` counter)
3. SqlAlchemyUnitOfWork(session_factory): __enter__ opens a session and builds a repository on it,
   commit() -> session.commit(), rollback() -> session.rollback(), __exit__ also closes the session.
4. Service: transfer(uow, from_id, to_id, amount):
       with uow:
           src, dst = uow.accounts.get(from_id), uow.accounts.get(to_id)
           src.balance -= amount ; if src.balance < 0: raise ValueError("insufficient funds")
           dst.balance += amount
           uow.commit()
   If anything raises before commit -> automatic rollback, balances unchanged.
Use a plain @dataclass Account(id: str, balance: float) and, for SQLAlchemy, a mapped AccountRow.
Self-check expects these names: transfer, InMemoryUnitOfWork(initial: dict[str, float]), make_sqlalchemy_uow(initial: dict[str, float]), balances(uow) -> dict

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
    for u in [InMemoryUnitOfWork({"a": 100, "b": 0}), make_sqlalchemy_uow({"a": 100, "b": 0})]:
        transfer(u, "a", "b", 40)
        assert balances(u) == {"a": 60, "b": 40}
        try:
            transfer(u, "a", "b", 1000)
            raise AssertionError("should raise")
        except ValueError:
            pass
        assert balances(u) == {"a": 60, "b": 40}, "failed transfer must roll back"
    print("All checks passed!")
