import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_data():
    """Load CIFAR-10 and preprocess"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    return trainset, testset

def prepare_federated_data(trainset, num_clients):
    """Split dataset for each client."""
    # Set different number of data points for each client (Non-IID)
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    # Allocate remaining data to the last client
    lengths[-1] += len(trainset) - sum(lengths)

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Create DataLoader for each client
    trainloaders = []
    for ds in datasets:
        trainloaders.append(DataLoader(ds, batch_size=32, shuffle=True))
    
    return trainloaders

# Test DataLoader
# _, testset = load_data()
# testloader = DataLoader(testset, batch_size=32)