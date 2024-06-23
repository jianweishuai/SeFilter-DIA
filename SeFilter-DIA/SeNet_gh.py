# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:15:59 2022

@author: GuoHuan
"""

import torch
from torch import nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filter3,filter3//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1*x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1

class senet(nn.Module):
    def __init__(self,cfg):
        super(senet,self).__init__()
        classes = cfg['classes']
        num = cfg['num']
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0],1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048,classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_channels, filters, num, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)


def Senet():
    cfg = {
        'num':(3,4,6,3),
        'classes': (10)
    }
    return senet(cfg)
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def load_dataset(batch_size):
    train_set = torchvision.datasets.CIFAR10(
        root="data/cifar-10", train=True,
        download=True, transform=transforms.ToTensor()
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data/cifar-10", train=False,
        download=True, transform=transforms.ToTensor()
    )
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return train_iter, test_iter


def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    net.train()
    record_train = list()
    record_test = list()

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            output = net(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), train_loss / (i + 1), \
                            train_acc, get_cur_lr(optimizer)))


        if lr_scheduler is not None:
            lr_scheduler.step()

        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:
            record_test.append(test(net, test_iter, criterion, device))
        record_train.append(train_acc)

    return record_train, record_test


def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)

            output = net(X)
            loss = criterion(output, y)

            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()

    test_acc = 100.0 * correct / total

    print("test_loss: {:.3f} | test_acc: {:6.3f}%"\
          .format(loss.item(), test_acc))
    print("************************************\n")
    net.train()

    return test_acc


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


import torch.optim as optim


BATCH_SIZE = 128
NUM_EPOCHS = 12
NUM_CLASSES = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    net = net = Senet()
    net = net.to(DEVICE)

    train_iter, test_iter = load_dataset(BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    record_train, record_test = train(net, train_iter, criterion, optimizer, \
          NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler, test_iter)

    learning_curve(record_train, record_test)


if __name__ == '__main__':
    main()

