# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:09:07 2022

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
import copy



individual = True
if individual:
    Ad = 'independent'
else:
    Ad = 'associated'
print('The distribution of the dataset is', Ad)
model_name = 'Transformer'
# batch_size = 128
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

'''
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
'''

class Config(object):
    def __init__(self):
        self.device = torch.device('cpu')
        # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dropout = 0.2
        self.batch_size = 128
        self.pad_size = 85
        self.embed = 6
        self.num_classes = 1
        self.learning_rate = 1e-2
        self.dim_model = 6
        self.hidden = 200
        self.last_hidden = 100
        self.num_head = 2
        self.num_encoder = 2
        
config = Config()
batch_size = config.batch_size

def data_loader(data, label):
    x, y = torch.from_numpy(data), torch.from_numpy(label)
    x, y = x.float(), y.float()
    dataset = TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle = True)
    return data_loader

train_data = np.load(os.getcwd() + '/data/train_data_gh.npy').reshape(-1,6,85).transpose(0,2,1)
train_label = np.load(os.getcwd() + '/data/train_label_gh.npy')
valid_data = np.load(os.getcwd() + '/data/valid_data_gh.npy').reshape(-1,6,85).transpose(0,2,1)
valid_label = np.load(os.getcwd() + '/data/valid_label_gh.npy')
test_data = np.load(os.getcwd() + '/data/test_data_gh.npy').reshape(-1,6,85).transpose(0,2,1)
test_label = np.load(os.getcwd() + '/data/test_label_gh.npy')

train_loader = data_loader(train_data, train_label)
valid_loader = data_loader(valid_data, valid_label)
test_loader = data_loader(test_data, test_label)


"""======================== 2. Model Building ========================"""


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:,0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:,1::2])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device) #nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去
        out = self.dropout(out)
        return out
        
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()
        
    def forward(self, Q, K, V, scale=None):
    
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head) #view()的作用相当于numpy中的reshape，重新定义矩阵的形状。
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
    
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out
    
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)
        
    def forward(self, x):
        out = self.attention(x.to(config.device))
        out = self.feed_forward(out)
        return out

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
         
        self.position_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)  
            for _ in range(config.num_encoder)])
        
        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.o = nn.Sigmoid()
        
    def forward(self, x):
        out = x
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.o(out)
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
                    
                    torch.save(net, os.getcwd() + '/param/' + model_name +'/result/' + '/net_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.pth')
                    torch.save(net.state_dict(), os.getcwd() + '/param/' + model_name + '/net_params_' + str(epoch) +'_'+ Ad +'_'+ '_epoch.path')
            
                    figure1 = plt.figure()
                    plt.title('ROC Curve'+'_AUC')
                    plt.plot(fpr, tpr, color='red',label='Training'+ str(train_auc))
                    plt.plot(fpr_valid, tpr_valid, color='green',label='Validatin' +str(valid_auc) )
                    plt.plot(fpr_test, tpr_test, color='blue',label='Testing' + str(test_auc))
                    plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad + "ROC.jpg")
                    
                    # figure2 = plt.figure()
                    # plt.title('Validating ROC Curve'+'_AUC'+ str(valid_auc))
                    # plt.plot(fpr_valid, tpr_valid, color='greed',label='Validating')
                    # plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad  +'_valid_'+ "ROC.jpg")
                    
                    # figure3 = plt.figure()
                    # plt.title('Testing ROC Curve'+'_AUC'+ str(test_auc))
                    # plt.plot(fpr_test, tpr_test, color='greed',label='Testing')
                    # plt.savefig('./figure/'+ model_name + '/' + model_name +'_'+ Ad  +'_test_'+ "ROC.jpg")
                    
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


# Params



batch_size = config.batch_size
BATCH_SIZE = batch_size
NUM_EPOCHS = 101
NUM_CLASSES = 1
LEARNING_RATE = 0.003
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_PRINT = 100
DEVICE = "cpu"
# "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    net = Model(config)

    criterion = nn.BCELoss()
    
    optimizer = getattr(optim, 'Adam')(net.parameters(), lr=0.01)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(net, train_loader, valid_loader, test_loader, criterion, optimizer, \
          NUM_EPOCHS, config.device, NUM_PRINT, lr_scheduler)

    # learning_curve(record_train, record_test)