from black import out
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1)
        self.drop_1 = nn.Dropout(0.25)
        self.drop_2 = nn.Dropout(0.5)
        self.fully_1 = nn.Linear(9216, 128)
        self.fully_2 = nn.Linear(128, 10)
        
    
    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop_1(x)
        x = torch.flatten(x, 1)
        x = self.fully_1(x)
        x = F.relu(x)
        x = self.drop_2(x)
        x = self.fully_2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch * len(data)} / {len(train_loader.dataset)}]")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Accuracy {correct}/{len(test_loader.dataset)}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    train_kwargs = { 'batch_size': 64 }
    test_kwargs = { 'batch_size': 1000 }

    if torch.cuda.is_available():
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform = Compose([
        ToTensor(),
        Normalize((0.1307, ), (0.3081, ))
    ])

    dataset_train = datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transform
    )

    dataset_test = datasets.MNIST(
        '../data',
        train=False,
        transform=transform
    )

    train_loader = DataLoader(dataset_train, **train_kwargs)
    test_loader = DataLoader(dataset_test, **test_kwargs)

    model = Network().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 15):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()