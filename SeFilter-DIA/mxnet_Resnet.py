# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 23:28:05 2022

@author: GuoHuan
"""


#%% mxnet

import os
import copy
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
from sklearn import metrics
from mxnet import gluon, init
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


"""======================== 1. Loading Data ========================"""
individual = False
if individual:
    Ad = 'independent'
else:
    Ad = 'associated'
print('The distribution of the dataset is', Ad)
model_name = 'convLSTM'
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


data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
data_iter_valid  = mx.io.NDArrayIter(valid_data, label=valid_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')


"""======================== 2. Model Building ====================s===="""
ctx = mx.gpu()

class Residual(nn.Block):  
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

Resnet = nn.Sequential()
Resnet.add(
    nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
    nn.BatchNorm(), nn.Activation('relu'),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

Resnet.add(
    resnet_block(64, 2, first_block=True),
    resnet_block(128, 2),
    resnet_block(256, 2),
    resnet_block(512, 2),
    nn.GlobalAvgPool2D(),
    nn.Dense(1, activation='sigmoid')
)




"""======================== 3. Training process ========================"""

def testing_auc(net, test_iter, loss, ctx):
    
    test_iter.reset()
    n = 0
    test_l_sum = 0.0
    
    predict_score = []
    target_label = []
	
    for batch in test_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        test_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predict_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    test_precison = precision_score(target_label, np.around(predict_score,0).astype(int), average='binary')
    test_recall = recall_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
    test_f1_score = f1_score(target_label, np.around(predict_score,0).astype(int), pos_label=1)
    test_acc = accuracy_score(target_label, np.around(predict_score,0).astype(int))
    test_confusionMatrix = confusion_matrix(target_label, np.around(predict_score,0).astype(int))
        
    
    # print('Model test_predict_label_count',len(predit_score))
    
    testing_loss = test_l_sum / n
    
    return testing_loss, test_auc, predict_score, fpr, tpr, test_precison, test_recall, test_f1_score, test_acc, test_confusionMatrix


def validating_auc(net, valid_iter, loss, ctx):
    
    n = 0
    valid_l_sum = 0.0
    
    predict_score = []
    target_label = []
	
    for batch in valid_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        valid_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predict_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
     
    validating_loss = valid_l_sum / n
    
    return validating_loss, valid_auc, predict_score, fpr, tpr



def train_ch5(net, train_iter, valid_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    
    loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    
    start = time.time()
    print('epoch'.center(20), 'training_loss'.center(20),
          'validating_loss'.center(20), 'valid_auc'.center(20))
    
    train_loss, valid_loss, test_auc = [], [], []
    best_valid_auc, best_test_auc, best_epoch = 0.0, 0.0, 0
    
    for epoch in range(num_epochs):
        
        n = 0.0
        train_l_sum = 0.0
        
        train_iter.reset()
        valid_iter.reset()
                
        predict_score = []
        target_label = []
        
        for batch in train_iter:
            
            X = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            
            l.backward()        #计算梯度
            trainer.step(batch_size)        #更新参数
            train_l_sum += l.mean().asscalar()        #.asscalar()将NDArray转换为标量
            n += 1
            
            target_label.extend(y.asnumpy().tolist())
            predict_score.extend(y_hat.asnumpy().tolist())
        
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predict_score, pos_label=1)
        train_auc = metrics.auc(fpr, tpr)
        
        training_loss = train_l_sum / n
        
        validating_loss, valid_auc, predict_score_valid, fpr_valid, tpr_valid = validating_auc(net, valid_iter, loss, ctx)
        
        train_loss.append(training_loss)
        valid_loss.append(validating_loss)
        

        print(str(epoch).center(20), str(training_loss).center(20),
              str(validating_loss).center(20), str(valid_auc).center(20))
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                testing_loss, test_auc, predict_score_test, fpr_test, tpr_test, test_precison, test_recall, test_f1_score, test_acc, test_confusionMatrix = testing_auc(net, test_iter, loss, ctx)
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_epoch = epoch
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
            
                        net.save_parameters(os.getcwd() + '/param/'+ model_name +'/result/' + model_name + '_cnn_epoch' + str(epoch) + '.mxnet')
                        
    print("The training_auc of the test_data Test_XICs.npy:", train_auc, "The validing_auc:", valid_auc, "The testing_auc:", test_auc)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title(model_name + '_Training Loss')
    x = np.linspace(0, len(train_loss), len(train_loss))
    ax.plot(x, train_loss, label='Training_loss')
    ax.plot(x, valid_loss, label='validating_loss')
    plt.savefig("./figure/convLSTM/" + model_name +'_'+ Ad +'_'+ "_trainingloss.jpg")
    
    return train_loss, valid_loss, testing_loss


#%%

##### Params

######
ctx = mx.gpu()
lr, num_epochs = 0.001, 201


if __name__ == '__main__':
    time_start = time.time()
    
    
    Resnet.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=mx.gpu())
    Resnet.cast('float32')
    #optim = optimizer.Adam(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,wd=1e-5)
    trainer = gluon.Trainer(Resnet.collect_params(), 'adam',optimizer_params={'learning_rate':lr,'wd':5e-4}) #{'wd':5e-4}
    train_loss, valid_loss, testing_loss = train_ch5(Resnet, data_iter_train, data_iter_valid, data_iter_test, batch_size, trainer, ctx, num_epochs)
    
    time_end = time.time()
    time_sum = time_end - time_start
    print(model_name, "_mxnet time:", time_sum)


