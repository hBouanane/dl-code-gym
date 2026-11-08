"""
Exercise 06 - Auth: password hashing, OAuth2 password flow, JWT  [security]

1. Password hashing with hashlib.pbkdf2_hmac("sha256", password, salt, 100_000)
   hash_password(pw) -> "salt_hex$hash_hex" ; verify_password(pw, stored) -> bool
   (in real projects use passlib/bcrypt or argon2; NEVER store plain passwords)
2. JWT with PyJWT: create_access_token(sub, minutes=30):
      payload = {"sub": sub, "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
      jwt.encode(payload, SECRET_KEY, algorithm="HS256")
3. oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
   POST /token takes form_data: OAuth2PasswordRequestForm = Depends(); checks USERS
   -> {"access_token": ..., "token_type": "bearer"} or 401
4. get_current_user(token = Depends(oauth2_scheme)): decode with jwt.decode(..., algorithms=["HS256"]),
   401 on jwt.InvalidTokenError (expired, bad signature...). Return the username.
5. GET /me -> {"username": current_user}
USERS = {"haitam": hash_password("pass1234")}
Self-check expects these names: app, hash_password, verify_password, create_access_token
"""

import hashlib
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "change-me-in-env"   # in real life: from settings / environment variable
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 100_000)
    return digest.hex() == digest_hex


def create_access_token(sub: str, minutes: int = 30) -> str:
    payload = {"sub": sub, "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


USERS = {"haitam": hash_password("pass1234")}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    stored = USERS.get(form_data.username)
    if not stored or not verify_password(form_data.password, stored):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(form_data.username), "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token",
                            headers={"WWW-Authenticate": "Bearer"})
    return payload["sub"]


@app.get("/me")
def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


# ============ SELF-CHECK (don't edit) ============
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    assert verify_password("abc12345", hash_password("abc12345"))
    assert not verify_password("wrong", hash_password("abc12345"))
    c = TestClient(app)
    assert c.post("/token", data={"username": "haitam", "password": "bad"}).status_code == 401
    tok = c.post("/token", data={"username": "haitam", "password": "pass1234"}).json()["access_token"]
    assert c.get("/me", headers={"Authorization": f"Bearer {tok}"}).json() == {"username": "haitam"}
    assert c.get("/me").status_code == 401
    assert c.get("/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    expired = create_access_token("haitam", minutes=-1)
    assert c.get("/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    print("All checks passed!")
