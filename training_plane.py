import torch
import torch.nn as nn
import torch.optim as optim
from mesh import *
from conv import *

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class SphereGaugeCNN(nn.Module):

    def __init__(self, rep):
        super().__init__()
        self.depth = len(rep)-1
        
        grid = SquareGrid(28)
        precomputation = KernelPrecomputation(EuclideanSpace(2), grid, grid.graphNeighbor(2), 4, 4, 0.5/28, 0.5/28, 3)
        self.samplingCoords = grid.samplingCoords()

        self.conv = []
        for d in range(self.depth):
            self.conv.append(SphereConv(rep[d], rep[d+1], precomputation))
        self.conv = nn.ModuleList(self.conv)
        self.fc = nn.Linear(self.conv[-1].dim_out, 10)
    
    def forward(self, x):
        # Input shape : len_grid x batch_size x 1
        # Output shape : batch_size x 10
        for d in range(self.depth):
            x = self.conv[d](x)
        x = torch.sum(x, dim=0)
        x = self.fc(x)
        return x

num_epoch = 10
batch_size = 32

# Normal MNIST (for planar CNN)
train_dataset = MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device))

def main():

    print('Precomputing filters')
    model = SphereGaugeCNN([{0:1}, {0:8,1:4,2:4,3:4}, {0:32}])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print()

    print('Training Start')
    step = 0
    for epoch in range(num_epoch):
        for x, y in train_loader:

            x, y_ = x.to(device), y.to(device)
            x = torch.reshape(x/255, (-1, 28, 28, 1))
            x = sampling(x, model.samplingCoords)

            optimizer.zero_grad()
            y = model(x)
            cost = criterion(y, y_)
            accuracy = 100/batch_size * torch.sum(torch.argmax(y, dim=1) == y_)
            cost.backward(retain_graph=True)
            optimizer.step()
            print('Step: ' + str(step))
            print('Cross entropy: ' + str(cost.item()))
            print('Accuracy: ' + str(accuracy.item()) + '%')
            print()

            del y, cost, accuracy
            torch.cuda.empty_cache()
            step += 1

main()