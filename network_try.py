import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32*26*26, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 102)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(-1, self.flat(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def flat(self, x):
        return x.shape[1]*x.shape[2]*x.shape[3]


def net_train(date_loader, epoch, criterion, device, net, optimizer):
    net.to(device)

    for i in range(epoch):
        loss_epoch = 0

        for batch_idx, (inputs, labels) in enumerate(date_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
        print('epoch : ' + str(i), 'train_loss = ' + str(100*loss_epoch/len(train_loader.dataset)))
        test(net, device, test_loader)


def test(net, device, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, target).item()
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('test_loss = ' + str(100*test_loss/len(test_loader.dataset)), 'Accuracy: {:.3f}%'.format(100.*correct/len(test_loader.dataset)))


if __name__ == '__main__':
    train_dir = 'oxford-102-flowers/train'
    test_dir = 'oxford-102-flowers/test'
    valid_dir = 'oxford-102-flowers/valid'

    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225)),
                                           ])

    test_valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                     std=(0.229, 0.224, 0.225)),
                                           ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=16)

    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    print(len(train_loader), len(test_loader))
    net_train(train_loader, 200, criterion, device, net, optimizer)

