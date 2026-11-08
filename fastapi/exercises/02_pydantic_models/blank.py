"""
Exercise 02 - Pydantic v2: request/response models, Field, validators  [basics]

1. class UserCreate(BaseModel): name: str = Field(min_length=2, max_length=50)
                                email: EmailStr is optional extra; use str + a @field_validator
                                  "email" that requires "@" (raise ValueError otherwise)
                                age: int = Field(ge=0, le=150)
                                password: str = Field(min_length=8)
2. class UserOut(BaseModel): id: int ; name: str ; email: str   (NO password!)
3. POST /users (status_code=201, response_model=UserOut): store in a dict DB with an auto id,
   return the stored user -> FastAPI filters out the password thanks to response_model.
4. Pydantic outside FastAPI:
     u = UserCreate(name="Haitam", email="h@x.com", age=25, password="secret123")
     u.model_dump() -> dict ; u.model_dump_json() -> str ; UserCreate.model_validate(dict) ;
     model_config = ConfigDict(str_strip_whitespace=True) to auto-strip strings
5. A @model_validator(mode="after") on a class PasswordChange(new: str, confirm: str) that
   raises if new != confirm.
Self-check expects these names: app, UserCreate, UserOut, PasswordChange

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
    from pydantic import ValidationError
    c = TestClient(app)
    r = c.post("/users", json={"name": "Haitam", "email": "h@x.com", "age": 25, "password": "secret123"})
    assert r.status_code == 201 and "password" not in r.json() and r.json()["id"] >= 1
    assert c.post("/users", json={"name": "H", "email": "h@x.com", "age": 25, "password": "secret123"}).status_code == 422
    assert c.post("/users", json={"name": "Haitam", "email": "nope", "age": 25, "password": "secret123"}).status_code == 422
    assert c.post("/users", json={"name": "Haitam", "email": "h@x.com", "age": -1, "password": "secret123"}).status_code == 422
    assert UserCreate(name=" Bob ", email="b@x.com", age=1, password="12345678").name == "Bob"
    try:
        PasswordChange(new="a", confirm="b")
        raise AssertionError("should fail")
    except ValidationError:
        pass
    print("All checks passed!")
