import torch
from src.model import BigramLanguageModel


def predictor(text) -> str:
    # load the model
    uri = "mlartifacts/927816015437864496/eea158f9f96c4164b0494e3c6ede2d80/artifacts/model/state_dict.pth"
    # load state dict
    state_dict = torch.load(uri)
    model = BigramLanguageModel(
        n_embd=384,
        block_size=256,
        n_head=4,
        n_layer=12,
        dropout=0.3,
        to_device="cpu",
    )
    model.load_state_dict(state_dict)
    encode = torch.tensor([model.encode(text)], dtype=torch.long)
    model.eval()
    return model.decode(model.generate(idx=encode, max_new_tokens=500)[0].tolist())


if __name__ == "__main__":
    text = "It was a dark and stormy night"
    print(predictor(text))
