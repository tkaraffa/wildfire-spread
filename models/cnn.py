import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=12, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.Sigmoid(),
            # torch.nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(),
            # nn.LogSoftmax()
            # nn.Flatten(),
            # nn.Linear(in_features=49152, out_features=4096),
        )

    def forward(self, x):
        return self.cnn(x)


def train(loader, model, loss_fn, optimizer, batch_size):
    size = len(loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(loader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
