from src.data import FyodorDataset
from src.model import BigramLanguageModel
import lightning.pytorch as pl
import torch


class BigramLightning(pl.LightningModule, BigramLanguageModel):
    def __init__(
        self,
        n_embd=32,
        block_size=8,
        n_head=4,
        n_layer=3,
        dropout=0.1,
        learning_rate=1e-3,
        to_device="cpu",
    ):
        pl.LightningModule.__init__(self)
        BigramLanguageModel.__init__(
            self, n_embd, block_size, n_head, n_layer, dropout, to_device
        )
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    from src.utils import get_files_from_folder, open_txt
    from torch.utils.data import DataLoader
    import uuid
    import mlflow
    from itertools import product
    import random

    torch.set_float32_matmul_precision("medium")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # create an experiment
    exp_name = str(uuid.uuid4()).split("-")[0]
    exp_id = mlflow.create_experiment(exp_name)

    # create grid of hyperparameters
    space_block_size = [512]
    space_n_embd = [768]
    space_n_head = [4]
    space_n_layer = [9]
    space_dropout = [0.1]
    space_learning_rate = [3e-4]

    # random grid
    params = list(
        product(
            space_block_size,
            space_n_embd,
            space_n_head,
            space_n_layer,
            space_dropout,
            space_learning_rate,
        )
    )
    random.shuffle(params)

    for block_size, n_embd, n_head, n_layer, dropout, learning_rate in params:
        with mlflow.start_run(experiment_id=exp_id) as run:
            artifact_path = "model"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = 64
            print("+++++++++++++++++++++++++++++++")
            print(
                {
                    "block_size": block_size,
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                }
            )

            mlflow.log_param("block_size", block_size)
            mlflow.log_param("n_embd", n_embd)
            mlflow.log_param("n_head", n_head)
            mlflow.log_param("n_layer", n_layer)
            mlflow.log_param("dropout", dropout)

            books = get_files_from_folder("books")
            books_string = [open_txt(f"books/{i}") for i in books]
            books = "\n".join(books_string)
            train_dataset = FyodorDataset(
                books[: int(len(books) * 0.8)],
                length=batch_size * 100,
                block_size=block_size,
                batch_size=batch_size,
            )
            val_dataset = FyodorDataset(
                books[int(len(books) * 0.8) :],
                length=batch_size * 10,
                block_size=block_size,
                batch_size=batch_size,
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=24
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, num_workers=24
            )

            trainer = pl.Trainer(
                accelerator="gpu", devices=1, max_epochs=200, log_every_n_steps=1
            )
            model = BigramLightning(
                to_device=device,
                block_size=block_size,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                learning_rate=learning_rate,
            )

            mlflow.pytorch.autolog()
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path)
