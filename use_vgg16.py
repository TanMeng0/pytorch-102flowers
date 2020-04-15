import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict     #有序字典
import torch.nn.functional as F
import seaborn as sb  #matplotlib的补充，可以用于作图，可视化数据

train_dir = './oxford-102-flowers/train'
test_dir = './oxford-102-flowers/test'
valid_dir = './oxford-102-flowers/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),    #随机裁剪成224*224是因为后面的VGG16需要这种输入大小
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean =(0.485, 0.456, 0.406),
                                                           std = (0.229, 0.224, 0.225))])  #(RGB-mean)/std，这里RGB已经除以255，范围[0,1]
                                     #为什么采用这种均值标准差进行归一化？Using the mean and std of Imagenet is a common practice.
                                    # They are calculated based on millions of images. If you want to train from scratch on your own dataset,
                                    # you can calculate the new mean and std. Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.

test_valid_transforms = transforms.Compose([transforms.Resize(256),   #如果参数为单个整数n，则将短边缩放到n，长边按原比例缩放
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean =(0.485, 0.456, 0.406),
                                                           std = (0.229, 0.224, 0.225))])

#使用预处理格式加载数据
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_valid_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = test_valid_transforms)

#创建三个加载器，分别为训练，验证，测试
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = 20)

fmodel = models.vgg16(pretrained = True)

for param in fmodel.parameters():
    param.require_grad = False

print(fmodel)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(4096, 1000)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(1000, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

fmodel.classifier = classifier
print(fmodel)

criterion = nn.NLLLoss()
optimizer = optim.Adam(fmodel.classifier.parameters(), lr=0.001)


def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    model.cuda()

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    print('the accuracy is {:.4f}'.format(correct/total))


def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(inputs.shape, outputs.shape, labels.shape, loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                print('EPOCHS : {}/{}'.format(e+1, epochs),
                      'Loss ：{:.4f}'.format(running_loss/print_every))
                accuracy_test(model, valid_loader)


deep_learning(fmodel, train_loader, 100, 40, criterion, optimizer, 'cuda')

















