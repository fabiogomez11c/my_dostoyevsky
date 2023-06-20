import torch
import mlflow.pytorch
from src.model import BigramLanguageModel


if __name__ == "__main__":
    # load the model
    uri = "mlartifacts/588780005033708639/f175e6f9bf7c48b48d2af27c396553f5/artifacts/model"
    state_dict = mlflow.pytorch.load_state_dict(uri)
    model = BigramLanguageModel(
        n_embd=384,
        block_size=128,
        n_head=2,
        n_layer=6,
        dropout=0.3,
        to_device="cpu",
    )
    model.load_state_dict(state_dict)

    x = torch.zeros((1, 1), dtype=torch.long)
    x[0, 0] = 1
    # x = x.to("cuda")
    # model = model.to("cuda")
    model.eval()
    print(model.decode(model.generate(idx=x, max_new_tokens=500)[0].tolist()))

    print("Done")
