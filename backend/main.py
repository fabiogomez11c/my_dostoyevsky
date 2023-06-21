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

model_dir = "model_store/attention_statedict.pth"
model = BigramLanguageModel(
    n_embd=384,
    block_size=256,
    n_head=4,
    n_layer=12,
    dropout=0.3,
    to_device="cpu",
)
fyodor = torch.load(model_dir)
model.load_state_dict(fyodor)
model.eval()


class Input(BaseModel):
    text: str


@app.post("/predict")
def predict(input: Input):
    encoded = torch.tensor([model.encode(input.text)], dtype=torch.long)
    encoded_generated = model.generate(encoded, 500)
    decode = model.decode(encoded_generated[0].tolist())
    return {"text": decode}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
