from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch import nn


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    model.eval()
    app.state.model = model
    yield


app = FastAPI(lifespan=lifespan)


class Features(BaseModel):
    values: list[float]


@app.post("/predict")
def predict(body: Features):
    with torch.inference_mode():
        logits = app.state.model(torch.tensor([body.values]))
        probs = torch.softmax(logits, dim=-1)[0]
    return {"class": int(probs.argmax()), "probs": [round(p, 4) for p in probs.tolist()]}


@app.get("/health")
def health():
    return {"status": "ok", "torch": torch.__version__}
