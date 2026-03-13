import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.lenet5 import LeNet5
from dataset.mnist import get_datasets
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

num_epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
workers = config["train dataset"]["num_workers"]


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _ = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LeNet5().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    for epoch in range(num_epochs):

        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss {loss.item()}")
    torch.save(model.state_dict(), "results/checkpoints/lenet5.pth")


if __name__ == "__main__":
    train()
