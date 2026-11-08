import os

from fastapi import FastAPI
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])
app = FastAPI()

with engine.begin() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS visits (id SERIAL PRIMARY KEY)"))


@app.post("/visits")
def add_visit():
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO visits DEFAULT VALUES"))
        total = conn.execute(text("SELECT COUNT(*) FROM visits")).scalar_one()
    return {"total_visits": total}
