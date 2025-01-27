import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net
from train import train
from utils import get_device

def main():
    device = get_device()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion, epoch)
    torch.save(model.state_dict(), "mnist_model.pth")

if __name__ == '__main__':
    main()
