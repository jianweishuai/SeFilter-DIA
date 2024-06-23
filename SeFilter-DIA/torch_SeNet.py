# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:14:58 2022

@author: GuoHuan
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:53:51 2022

@author: GuoHuan
"""


#%% pytorch
import os
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import argparse
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score



individual = False
if individual:
    Ad = 'independent'
else:
    Ad = 'associated'
print('The distribution of the dataset is', Ad)
model_name = 'Senet'
batch_size = 256
"""======================== 1. Loading Data ========================"""
if individual:
    train_Data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
    train_Label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')
    test_data = np.load(os.getcwd() + '/data/Test_XICs.npy')
    test_label = np.load(os.getcwd() + '/data/Test_labels.npy')
    train_data, valid_data, train_label, valid_label = train_test_split(train_Data, train_Label, stratify=train_Label,
                                                                        test_size=0.3, random_state=5)

else:
    train_data = np.load(os.getcwd() + '/data/train_data_gh.npy')
    train_label = np.load(os.getcwd() + '/data/train_label_gh.npy')
    valid_data = np.load(os.getcwd() + '/data/valid_data_gh.npy')
    valid_label = np.load(os.getcwd() + '/data/valid_label_gh.npy')
    test_data = np.load(os.getcwd() + '/data/test_data_gh.npy')
    test_label = np.load(os.getcwd() + '/data/test_label_gh.npy')


data1_ = []
for i in range(len(train_data)):
    data1_.append(preprocessing.minmax_scale(np.array(train_data[i]).reshape(6, 85), axis=1))
train_data = np.array(data1_).reshape(len(data1_), 1, 6, 85)

data2_ = []
for i in range(len(valid_data)):
    data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6, 85), axis=1))
valid_data = np.array(data2_).reshape(len(data2_), 1, 6, 85)

data3_ = []
for i in range(len(test_data)):
    if individual:
        data3_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6, 85), axis=1))
    else:
        data3_.append(preprocessing.minmax_scale(np.array(test_data[i]).reshape(6, 85), axis=1))
test_data = np.array(data3_).reshape(len(data3_), 1, 6, 85)

test_label = np.squeeze(test_label)

x_train, y_train = torch.from_numpy(train_data), torch.from_numpy(train_label)
x_valid, y_valid = torch.from_numpy(valid_data), torch.from_numpy(valid_label)
x_test, y_test = torch.from_numpy(test_data), torch.from_numpy(test_label)

x_train, x_valid, x_test = x_train.float(), x_valid.float(), x_test.float()
y_train, y_valid, y_test = y_train.float(), y_valid.float(), y_test.float()

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


"""======================== 2. Model Building ========================"""

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
            # 二维卷积，第一个是通道数1， 64是batch_size
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # 批量归一化， 64是batch_size
            nn.BatchNorm2d(64),
            # 激活函数为ReLu： f(x) = max(0,x)
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0],1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
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


def Model():
    cfg = {
        'num':(3,4,6,3),
        'classes': (1)
    }
    return senet(cfg)



"""======================== 3. Training process ========================"""
def train(net, train_iter, valid_iter, test_iter, criterion, optimizer, num_epochs, device, num_print,
          lr_scheduler=None):
    
    net.train()
    start = time.time()
    print('epoch'.center(20), 'training_loss'.center(20),
          'validating_loss'.center(20), 'valid_auc'.center(20))
    
    best_valid_auc, best_test_auc, best_epoch = 0.0, 0.0, 0
    trainLoss, validLoss, testAuc = [], [], []
    
    for epoch in range(num_epochs):

        train_loss, training_loss = 0.0, 0.0

        predict_score = []
        target_label = []

        for i, (X, y) in enumerate(train_iter):

            X, y = X.to(device), y.to(device)
            output = net(X).squeeze().to(torch.float32)
            # if i == 0:
            #     print(X.shape, y.shape)
            #     print(output.shape, y.shape)
            #     print(output.type, y.type)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            target_label.extend(y.tolist())
            predict_score.extend(output.tolist())

        fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
        train_auc = metrics.auc(fpr, tpr)

        training_loss = train_loss / (i + 1)
        trainLoss.append(training_loss)
        
        validating_loss, valid_auc, predict_score_valid, fpr_valid, tpr_valid = valid(net, valid_iter, criterion, device)
        validLoss.append(validating_loss)
        
        print(str(epoch).center(20), str(training_loss).center(20),
              str(validating_loss).center(20), str(valid_auc).center(20))
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                testing_loss, test_auc, predict_score_test, fpr_test, tpr_test, test_precison, test_recall, test_f1_score, test_confusionMatrix, test_acc = test(net, test_iter, criterion, device)
                # testAuc.append(test_auc)
                
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_epoch = epoch
                    
                    torch.save(net, os.getcwd() + '/param/' + model_name + '/net_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.pth')
                    torch.save(net.state_dict(), os.getcwd() + '/param/' + model_name + 'net_params_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.path')
            
                    figure1 = plt.figure()
                    plt.title('ROC Curve'+'_AUC'+ str(test_auc))
                    plt.plot(fpr_test, tpr_test)
                    plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad  +'_'+ "ROC.jpg")
                    
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_distributuin_test.txt', predict_score_test, fmt='%f', delimiter=',')
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_test_fpr', fpr_test, fmt='%f', delimiter=',')
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_test_tpr', tpr_test, fmt='%f', delimiter=',')
                    
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_distributuin_valid.txt', predict_score_valid, fmt='%f', delimiter=',')
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_valid_fpr', fpr_valid, fmt='%f', delimiter=',')
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_valid_tpr', tpr_valid, fmt='%f', delimiter=',') 
        
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_distributuin_train.txt', predict_score, fmt='%f', delimiter=',')                
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_train_fpr', fpr, fmt='%f', delimiter=',')
                    np.savetxt(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_roc_train_tpr', tpr, fmt='%f', delimiter=',')
                
                    
                    out_file = os.getcwd() + '/figure/'+ model_name + '/' + Ad +'_'+'importantCriteria.txt'
                    with open(out_file, 'w') as out_file:
                        
                        out_str = ""
                        out_str = "Best Epoch is :\t" 
                        out_str += str(best_epoch)
                        out_str += "\nBest_test_auc:\t" 
                        out_str += str(best_test_auc)
                        out_str += "\ntrain_auc:\t" 
                        out_str += str(train_auc)
                        out_str += "\nvalid_auc:\t" 
                        out_str += str(best_valid_auc)
                        out_str += "\nprecision:\t" 
                        out_str += str(test_precison)
                        out_str += "\nrecall:\t" 
                        out_str += str(test_recall)
                        out_str += "\nfi_score:\t" 
                        out_str += str(test_f1_score)
                        out_str += "\nAcc:\t" 
                        out_str += str(test_acc)
                        
                        out_file.write(out_str)
        
            
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title(model_name + '_Training Loss')
    x = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS)
    ax.plot(x, trainLoss, label='Training_loss')
    ax.plot(x, validLoss, label='validating_loss')
    plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad +'_'+ "_trainingloss.jpg")
    
    # figure = plt.figure()
    # plt.title('Testing AUC')
    # x = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS)
    # plt.plot(x, testAuc)
    # plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad +'_'+  "_TestingAuc.jpg")
    
    print('best_test_auc_epoch:', best_epoch, 'best_test_auc:', best_test_auc)

    print("--- Whole cost time: {:.4f}s ---".format(time.time() - start))

    return training_loss, validating_loss, testing_loss, train_auc, valid_auc, test_auc


def valid(net, valid_iter, criterion, device):
    validating_loss, valid_loss = 0.0, 0.0
    net.eval()

    predict_score = []
    target_label = []

    with torch.no_grad():

        for i, (X, y) in enumerate(valid_iter):
            X, y = X.to(device), y.to(device)

            output = net(X).squeeze().to(torch.float32)
            # if i == 0:
                # print(X.shape, y.shape)
                # print(output.shape, y.shape)
                # print(output.type, y.type)
            loss = criterion(output, y)

            target_label.extend(y.tolist())
            predict_score.extend(output.tolist())

            valid_loss += loss.item()

        fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
        valid_auc = metrics.auc(fpr, tpr)

        validating_loss = valid_loss / (i + 1)

    net.train()

    return validating_loss, valid_auc, predict_score, fpr, tpr


def test(net, test_iter, criterion, device):
    test_loss, testing_loss = 0.0, 0.0
    net.eval()

    predict_score = []
    target_label = []

    with torch.no_grad():

        for i, (X, y) in enumerate(test_iter):
            X, y = X.to(device), y.to(device)

            output = net(X).squeeze().to(torch.float32)
            # if i == 0:
                # print(X.shape, y.shape)
                # print(output.shape, y.shape)
                # print(output.type, y.type)

            loss = criterion(output, y)
            test_loss += loss.item()

            target_label.extend(y.tolist())
            predict_score.extend(output.tolist())
        
        # print(predit_score)
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
        test_auc = metrics.auc(fpr, tpr)
        test_precison = precision_score(target_label, np.around(predict_score,0).astype(int), average='binary')
        test_recall = recall_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
        test_f1_score = f1_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
        test_acc = accuracy_score(target_label, np.around(predict_score,0).astype(int))
        test_confusionMatrix = confusion_matrix(target_label, np.around(predict_score,0).astype(int))
        # pre, rec, f1, sup = precision_recall_fscore_support(target_label, predit_score)
        # print("precision:", pre, "\nrecall:", rec, "\nf1-score:", f1, "\nsupport:", sup)

        testing_loss = test_loss / (i + 1)

    # print("test_loss: {:.3f} | test_auc: {:6.3f}%" \
    #       .format(loss.item(), test_auc * 100))
    # print("************************************\n")
    net.train()

    return testing_loss, test_auc, predict_score, fpr, tpr, test_precison, test_recall, test_f1_score, test_confusionMatrix, test_acc


if __name__ == '__main__':
    
    
    NUM_EPOCHS = 201
    
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    NUM_PRINT = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Model()
    net = net.to(DEVICE)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(net, train_loader, valid_loader, test_loader, criterion, optimizer,\
          NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler)

