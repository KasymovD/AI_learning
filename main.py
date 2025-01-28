import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from train import train
from utils import get_device

def main():
    device = get_device()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, criterion, epoch)
    torch.save(model.state_dict(), "cifar10_model.pth")

if __name__ == '__main__':
    main()
