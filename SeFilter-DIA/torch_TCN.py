# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:17:08 2022

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
model_name = 'TCN'
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
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(Model, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.out = nn.Sigmoid()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        o = self.out(o)
        return o



"""======================== 3. Training process ========================"""


def train(net, train_iter, valid_iter, test_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None):
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
            X = X.view(-1, input_channels, seq_length)
            
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
                
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_epoch = epoch
                    
                    torch.save(net, os.getcwd() + '/param/' + model_name + '/net_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.pth')
                    torch.save(net.state_dict(), os.getcwd() + '/param/' + model_name + 'net_params_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.path')
            
                    figure1 = plt.figure()
                    plt.title('ROC Curve'+'_AUC')
                    plt.plot(fpr, tpr, color='red',label='Training'+ str(train_auc))
                    plt.plot(fpr_valid, tpr_valid, color='green',label='Validatin' +str(best_valid_auc) )
                    plt.plot(fpr_test, tpr_test, color='blue',label='Testing' + str(best_test_auc))
                    plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad + "ROC.jpg")
                    
                    
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
        
        
    
    print("------------ Whole cost time: {:.4f}s ------------".format(time.time() - start))
        
    return training_loss, validating_loss, testing_loss, train_auc, valid_auc, test_auc



def valid(net, valid_iter, criterion, device):
    valid_loss, validating_loss = 0.0, 0.0
    net.eval()
    
    predict_score = []
    target_label = []
    
    with torch.no_grad():
        
        for i, (X, y) in enumerate(valid_iter):
            X, y = X.to(device), y.to(device)
            X = X.view(-1, input_channels, seq_length)  
            output = net(X).squeeze().to(torch.float32)
            # if i == 0:
            #     print(X.shape, y.shape)
            #     print(output.shape, y.shape)
            #     print(output.type, y.type)
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
            X = X.view(-1, input_channels, seq_length)
            # if permute:
            #     X = X[:, :, permute]
            output = net(X).squeeze().to(torch.float32)
            # if i == 0:
            #     print(X.shape, y.shape)
            #     print(output.shape, y.shape)
            #     print(output.type, y.type)
                
            loss = criterion(output, y)
            test_loss += loss.item()
            
            target_label.extend(y.tolist())
            predict_score.extend(output.tolist())
        
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
        test_auc = metrics.auc(fpr, tpr)
        
        test_precison = precision_score(target_label, np.around(predict_score,0).astype(int), average='binary')
        test_recall = recall_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
        test_f1_score = f1_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
        test_acc = accuracy_score(target_label, np.around(predict_score,0).astype(int))
        test_confusionMatrix = confusion_matrix(target_label, np.around(predict_score,0).astype(int))
        
        testing_loss = test_loss / (i + 1)
        
    net.train()
    return testing_loss, test_auc, predict_score, fpr, tpr, test_precison, test_recall, test_f1_score, test_confusionMatrix, test_acc


if __name__ == '__main__':
    
    n_classes = 1
    input_channels = 1
    seq_length = int(6*85 / input_channels)
    steps = 0
    kernel_size = 7
    dropout = 0.05
    nhid = 25
    levels = 8
    channel_sizes = [nhid] * levels
    
    
    BATCH_SIZE = batch_size
    NUM_EPOCHS = 101
    NUM_CLASSES = 1
    LEARNING_RATE = 0.03
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    NUM_PRINT = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Model(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    net = net.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = getattr(optim, 'Adam')(net.parameters(), lr=0.01)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    permute = torch.Tensor(np.random.permutation(6*85).astype(np.float64)).long()
    
    train(net, train_loader, valid_loader, test_loader, criterion, optimizer,\
          NUM_EPOCHS, DEVICE, NUM_PRINT, lr_scheduler)