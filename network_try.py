import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn6 = nn.BatchNorm2d(512)
        #
        # self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn7 = nn.BatchNorm2d(512)
        # self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn8 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 102)
        #self.fc2 = nn.Linear(1000, 500)
       # self.fc3 = nn.Linear(500, 102)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))), 2)
        # x = F.max_pool2d(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))), 2)
        # x = F.max_pool2d(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(x)))))), 2)

        x = self.avgpool(x)
        x = x.view(-1, self.flat(x))
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.2)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.2)
        # x = self.fc3(x)
        return x

    def flat(self, x):
        return x.shape[1]*x.shape[2]*x.shape[3]


def net_train(date_loader, epoch, criterion, device, net, optimizer, valid_loader):
    max_accuracy = 0
    min_loss = 9999
    accout = 0
    current_par_epoch = 0
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=1.1)
    scheduler2 = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
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
        print('epoch : ' + str(i+1), 'train_loss = ' + str(100*loss_epoch/len(date_loader.dataset)))
        accuracy, valid_loss = test(net, device, valid_loader)
        print('valid_loss = ' + str(valid_loss), 'Accuracy: {:.3f}%'.format(accuracy))

        accout += 1
        if valid_loss < min_loss or accuracy > max_accuracy:
            min_loss = valid_loss
            accout = 0
            torch.save(net, './model.pth')
            current_par_epoch = epoch
            scheduler.step()
        else:
            net = torch.load('./model.pth')
            print('当前迭代参数：' + str(current_par_epoch))
            scheduler2.step()


        if accout > 200:                  #如果200个epoch正确率没有更高则停止迭代
            print('model accuracy : {:.3f}%'.format(max_accuracy))
            print('valid_loss : '+str(valid_loss))
            break


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

    return 100.*correct/len(test_loader.dataset), 100*test_loss/len(test_loader.dataset)


if __name__ == '__main__':
    train_dir = 'oxford-102-flowers/train'
    test_dir = 'oxford-102-flowers/test'
    valid_dir = 'oxford-102-flowers/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16)

    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    net.to(device)
    print(len(train_loader), len(valid_loader), len(test_loader))
    net_train(train_loader, 20000, criterion, device, net, optimizer, valid_loader)
