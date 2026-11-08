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
