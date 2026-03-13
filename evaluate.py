import torch
from torch.utils.data import DataLoader

from models.lenet5 import LeNet5
from dataset.mnist import get_datasets
import os
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["testing"]["batch_size"]
workers = config["test dataset"]["num_workers"]


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataset = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LeNet5()
    model.load_state_dict(torch.load("results/checkpoints/lenet5.pth"))
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy:", 100 * correct / total)


if __name__ == "__main__":
    evaluate()
