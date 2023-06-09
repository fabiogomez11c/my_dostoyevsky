from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.model import BigramLanguageModel
import torch

app = FastAPI()

# allow CORS all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = BigramLanguageModel()
fyodor = torch.load("model_store/fyodor.pt")
model.load_state_dict(fyodor["state_dict"])


class Input(BaseModel):
    text: str


@app.post("/predict")
def predict(input: Input):
    encoded = torch.tensor([[model.encode(input.text)[-1]]], dtype=torch.long)
    encoded_generated = model.generate(encoded, 100)
    decode = model.decode(encoded_generated[0].tolist())
    return {"text": decode}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
