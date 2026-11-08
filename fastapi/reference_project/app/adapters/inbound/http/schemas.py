from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BookIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=200)


class BorrowIn(BaseModel):
    member: str = Field(min_length=1)


class BookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    author: str
    is_available: bool
    borrower: str | None
    borrowed_at: datetime | None
