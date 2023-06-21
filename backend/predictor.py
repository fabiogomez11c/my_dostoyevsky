import torch
import mlflow.pytorch
from src.model import BigramLanguageModel
from trainer import BigramLightning


if __name__ == "__main__":
    # load the model
    uri = "mlartifacts/927816015437864496/fcaa846067ef4c65a4e1670d3d95e003/artifacts/model"
    state_dict = mlflow.pytorch.load_state_dict(uri)
    model = BigramLightning(
        n_embd=192,
        block_size=256,
        n_head=8,
        n_layer=6,
        dropout=0.3,
        to_device="cpu",
    )
    model.load_state_dict(state_dict)

    x = torch.zeros((1, 1), dtype=torch.long)
    x[0, 0] = 1
    model.eval()
    print(model.decode(model.generate(idx=x, max_new_tokens=500)[0].tolist()))
