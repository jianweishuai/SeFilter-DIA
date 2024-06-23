# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:09:00 2022

@author: GuoHuan
"""


# %% pytorch
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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

individual = True
if individual:
    Ad = 'independent'
else:
    Ad = 'associated'
print('The distribution of the dataset is', Ad)
model_name = 'Width_torch_Resnet'
batch_size = 128

figure_folder = os.getcwd() + '/figure/' + model_name
param_folder = os.getcwd() + '/param/' + model_name
result_folder = param_folder + '/result'
if not os.path.exists(figure_folder):
    os.mkdir(figure_folder) 
    
if not os.path.exists(param_folder):
    os.mkdir(param_folder)
    
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
    


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

#%% ADD
Width = 41
halfWidth = Width // 2

def Width(Data, lenth, halfWidth):
    Data_after = []
    
    Data =Data.reshape(len(Data), lenth,85)
    
    for idx, data1 in enumerate(Data):
        xics = []
        for xic in data1:
            mid = len(xic) // 2
            left = mid - halfWidth
            right = mid + halfWidth + 1
            if left < 0:
                left = 0
            if right > len(xic) - 1:
                right = len(xic) -1
                
            xic = np.array(xic[left : right])
            xics.append(xic)
        Data_after.append(xics)
    return Data_after

        
train_data = np.array(Width(train_data, 6, halfWidth))
valid_data = np.array(Width(valid_data, 6, halfWidth))
if individual:
    test_data = np.array(Width(test_data, 7, halfWidth))
else:
    test_data = np.array(Width(test_data, 6, halfWidth))

#%%

data1_ = []
for i in range(len(train_data)):
    
    data1_.append(preprocessing.minmax_scale(np.array(train_data[i]).reshape(6, 41), axis=1))
train_data = np.array(data1_).reshape(len(data1_), 1, 6, 41)

data2_ = []
for i in range(len(valid_data)):
    data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6, 41), axis=1))
valid_data = np.array(data2_).reshape(len(data2_), 1, 6, 41)

data3_ = []
for i in range(len(test_data)):
    if individual:
        data3_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6, 41), axis=1))
    else:
        data3_.append(preprocessing.minmax_scale(np.array(test_data[i]).reshape(6, 41), axis=1))
test_data = np.array(data3_).reshape(len(data3_), 1, 6, 41)


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
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(ResBlock, self).__init__()
        #这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride[1] != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=1):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=(1,1))
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=(1,2))
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=(1,2))
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=(1,2))
        self.fc = nn.Linear(4032, num_classes)
        self.out = nn.Sigmoid()

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [(1,1)] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.out(out)
        return out

"""======================== 3. Training process ========================"""

def train(net, train_iter, valid_iter, test_iter, criterion, optimizer, num_epochs, device, num_print,
          lr_scheduler=None):
    
    net.train()
    start = time.time()
    print('epoch'.center(20), 'training_loss'.center(20),
          'validating_loss'.center(20), 'valid_auc'.center(20))
    
    best_train_auc, best_valid_auc, best_test_auc, best_epoch = 0.0, 0.0, 0.0, 0
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
                    best_train_auc = train_auc
                    best_valid_auc = valid_auc
                    
                    torch.save(net, param_folder + '/net_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.pth')
                    torch.save(net.state_dict(), os.getcwd() + '/param/' + model_name + 'net_params_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.path')
            
                    figure1 = plt.figure()
                    plt.title('ROC Curve'+'_AUC')
                    plt.plot(fpr, tpr, color='red',label='Training'+ str(train_auc))
                    plt.plot(fpr_valid, tpr_valid, color='green',label='Validatin' +str(valid_auc) )
                    plt.plot(fpr_test, tpr_test, color='blue',label='Testing' + str(test_auc))
                    plt.savefig(figure_folder + '/' + model_name +'_'+ Ad + "ROC.jpg")
                    
                    # figure2 = plt.figure()
                    # plt.title('Validating ROC Curve'+'_AUC'+ str(valid_auc))
                    # plt.plot(fpr_valid, tpr_valid, color='greed',label='Validating')
                    # plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad  +'_valid_'+ "ROC.jpg")
                    
                    # figure3 = plt.figure()
                    # plt.title('Testing ROC Curve'+'_AUC'+ str(test_auc))
                    # plt.plot(fpr_test, tpr_test, color='greed',label='Testing')
                    # plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad  +'_test_'+ "ROC.jpg")
                    
                    np.savetxt(result_folder + '/' + model_name + '_distributuin_test.txt', predict_score_test, fmt='%f', delimiter=',')
                    np.savetxt(result_folder + '/' + model_name + '_roc_test_fpr', fpr_test, fmt='%f', delimiter=',')
                    np.savetxt(result_folder + '/' + model_name + '_roc_test_tpr', tpr_test, fmt='%f', delimiter=',')
                    
                    np.savetxt(result_folder + '/' + model_name + '_distributuin_valid.txt', predict_score_valid, fmt='%f', delimiter=',')
                    np.savetxt(result_folder + '/' + model_name + '_roc_valid_fpr', fpr_valid, fmt='%f', delimiter=',')
                    np.savetxt(result_folder + '/' + model_name + '_roc_valid_tpr', tpr_valid, fmt='%f', delimiter=',') 
    
                    np.savetxt(result_folder + '/' + model_name + '_distributuin_train.txt', predict_score, fmt='%f', delimiter=',')                
                    np.savetxt(result_folder + '/' + model_name + '_roc_train_fpr', fpr, fmt='%f', delimiter=',')
                    np.savetxt(result_folder + '/' + model_name + '_roc_train_tpr', tpr, fmt='%f', delimiter=',')
                
                    
                    out_file = os.getcwd() + '/figure/'+ model_name + '/' + Ad +'_'+'importantCriteria.txt'
                    with open(out_file, 'w') as out_file:
                        
                        out_str = ""
                        out_str = "Best Epoch is :\t" 
                        out_str += str(best_epoch)
                        out_str += "\nBest_test_auc:\t" 
                        out_str += str(best_test_auc)
                        out_str += "\ntrain_auc:\t" 
                        out_str += str(best_train_auc)
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
    plt.savefig(figure_folder + '/' + model_name +'_'+ Ad +'_'+ "_trainingloss.jpg")
    
    # figure = plt.figure()
    # plt.title('Testing AUC')
    # x = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS)
    # plt.plot(x, testAuc)
    # plt.savefig(figure_folder + '/' + model_name +'_'+ Ad +'_'+  "_TestingAuc.jpg")
    
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


# Params

batch_size = batch_size
BATCH_SIZE = batch_size
NUM_EPOCHS = 101
NUM_CLASSES = 1
LEARNING_RATE = 0.003
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    net = ResNet(ResBlock)
    net = net.to(DEVICE)

    criterion = nn.BCELoss()
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
    #     weight_decay=WEIGHT_DECAY, nesterov=True)
    optimizer = getattr(optim, 'Adam')(net.parameters(), lr=0.01)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(net, train_loader, valid_loader, test_loader, criterion, optimizer, \
          NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler)

    # learning_curve(record_train, record_test)
    