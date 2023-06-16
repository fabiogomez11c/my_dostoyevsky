import time
import numpy as np
import pandas as pd
from utils import moveTo
from tqdm import tqdm
from tqdm import trange
import torch


def run_epoch(
    model,
    optimizer,
    data_loader,
    loss_func,
    device,
    results,
    score_funcs,
    prefix="",
):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs.
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model.
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary.
    desc -- a description to use for the progress bar.
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    pbar = tqdm(data_loader)
    counter = 0
    for inputs, labels in pbar:
        # Move the batch to the device we are using.
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)  # this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # add to preself(idx)ictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

        # update progress bar
        pbar.set_description(f"loss: {loss.item():.4f}")
        counter += 1

        if counter >= 32:
            break
    # end training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    if (
        len(y_pred.shape) == 2 and y_pred.shape[1] > 1
    ):  # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end - start  # time spent on epoch


def train_simple_network(
    model,
    loss_func,
    train_loader,
    test_loader=None,
    score_funcs=None,
    epochs=50,
    device="cpu",
    checkpoint_file=None,
    lr=0.001,
):
    """Train simple neural networks

    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs.
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training

    """
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    total_train_time = 0  # How long have we spent in the training loop?
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # SGD is Stochastic Gradient Decent.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    t = trange(epochs, desc="epoch 1")
    for epoch in t:
        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(
            model,
            optimizer,
            train_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix="train",
        )

        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)

        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(
                    model,
                    optimizer,
                    test_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="test",
                    desc="Testing",
                )

        # update progress bar
        t.set_description(f"epoch {epoch + 1} | loss: {results['train loss'][-1]:.4f}")

    if checkpoint_file is not None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "results": results,
            },
            checkpoint_file,
        )

    return pd.DataFrame.from_dict(results)


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    import torch.nn as nn
    from data import FyodorDataset
    from model import BigramLanguageModel
    from utils import get_files_from_folder, open_txt

    books = get_files_from_folder("books")
    books_string = [open_txt(f"books/{i}") for i in books]
    books = "\n".join(books_string)

    train_dataset = FyodorDataset(books[: int(len(books) * 0.8)])
    val_dataset = FyodorDataset(books[int(len(books) * 0.8) :])

    model = BigramLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    t = trange(10000)
    for steps in t:
        # sample a batch of data
        xb, yb = get_batch(train_dataset.data, block_size=8, batch_size=32)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t.set_description(f"loss: {loss.item():.4f}")

    print("Done")
