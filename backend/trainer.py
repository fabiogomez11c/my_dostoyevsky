from src.data import FyodorDataset
from src.model import BigramLanguageModel
import lightning.pytorch as pl
import torch


class BigramLightning(pl.LightningModule, BigramLanguageModel):
    def __init__(
        self, n_embd=32, block_size=8, n_head=4, n_layer=3, dropout=0.1, to_device="cpu"
    ):
        pl.LightningModule.__init__(self)
        BigramLanguageModel.__init__(
            self, n_embd, block_size, n_head, n_layer, dropout, to_device
        )

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    from src.utils import get_files_from_folder, open_txt
    from torch.utils.data import DataLoader

    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run() as run:
        batch_size = 256
        block_size = 8
        max_iters = 10000
        eval_iters = 200
        eval_interval = 500
        learning_rate = 1e-3
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_embd = 384
        n_head = 6
        n_layer = 6
        dropout = 0.4

        books = get_files_from_folder("books")
        books_string = [open_txt(f"books/{i}") for i in books]
        books = "\n".join(books_string)
        train_dataset = FyodorDataset(
            books[: int(len(books) * 0.8)],
            length=batch_size * 10,
            block_size=block_size,
            batch_size=batch_size,
        )
        val_dataset = FyodorDataset(
            books[int(len(books) * 0.8) :],
            length=batch_size * 100,
            block_size=block_size,
            batch_size=batch_size,
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=24
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=24)

        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_epochs=30, log_every_n_steps=1
        )
        model = BigramLightning(
            to_device=device,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
        )

        mlflow.pytorch.autolog()
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        x = torch.zeros((1, 1), dtype=torch.long)
        x[0, 0] = 1
        x = x.to("cuda")
        model = model.to("cuda")
        model.eval()
        print(model.decode(model.generate(idx=x, max_new_tokens=500)[0].tolist()))
