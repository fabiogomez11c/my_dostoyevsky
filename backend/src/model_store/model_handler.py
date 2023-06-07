import torch
import logging
from ts.torch_handler.base_handler import BaseHandler


class BigramLanguageModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def preprocess(self, data):
        return torch.tensor(data[0]["body"]["data"]), data[0]["body"]["max_token"]

    def inference(self, data):
        idx, max_token = data
        model = self.model
        output = model.decode(model.generate(idx, max_token)[0].tolist())
        return output

    def postprocess(self, data):
        return [data]
