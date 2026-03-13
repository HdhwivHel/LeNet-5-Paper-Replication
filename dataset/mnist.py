import torchvision.transforms.v2 as transforms
from torchvision.datasets import MNIST
import torch


def get_datasets():

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset
