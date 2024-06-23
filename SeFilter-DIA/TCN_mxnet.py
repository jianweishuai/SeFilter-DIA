# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:17:00 2022

@author: GuoHuan
"""
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


#%% casual convolution
class Chomp1d(gluon.nn.Block):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

    
# Res module
class TemporalBlock(gluon.nn.Block):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1D(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = mx.ndarray.Activation(act_type='relu')
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1D(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = mx.ndarray.Activation(act_type='relu')
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.HybridSequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                       self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1D(n_inputs, kernel_size=1, activation='relu') if n_inputs != n_outputs else None
        self.relu = mx.ndarray.Activation(act_type='relu')
        

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    
class TemporalConvNet(gluon.nn.Block):
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
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(layers)
        net.hybridize()
        self.network = net

    def forward(self, x):
        return self.network(x)
    

class Model(gluon.nn.Block):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(Model, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)
        self.linear = gluon.nn.Dense(output_size, activation='softmax')

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return mx.ndarray.log_softmax(o, dim=1)



#%% training processing
def testing_auc(net, test_iter, loss, ctx):
    
    test_iter.reset()
    n = 0
    test_l_sum = 0.0
    
    predit_score = []
    target_label = []
	
    for batch in test_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        test_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predit_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    print('TCN_mxnet test_predict_label_count',len(predit_score))
    # np.savetxt(os.getcwd() + '/202203/result_combined/TCN_mxnet_distributuin_test.txt', predit_score, fmt='%f', delimiter=',')
    # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_test_fpr', fpr, fmt='%f', delimiter=',')
    # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_test_tpr', tpr, fmt='%f', delimiter=',')
    
    
    
    testing_loss = test_l_sum / n
    
    return testing_loss, test_auc


def validating_auc(net, valid_iter, loss, ctx):
    
    n = 0
    valid_l_sum = 0.0
    
    predit_score = []
    target_label = []
	
    for batch in valid_iter:
            
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
    
        y_hat = net(X)
        l = loss(y_hat, y)
        
        n += 1
        valid_l_sum += l.mean().asscalar()
        
        target_label.extend(y.asnumpy().tolist())
        predit_score.extend(y_hat.asnumpy().tolist())
    
    fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    # np.savetxt(os.getcwd() + '/202203/result_combined/TCN_mxnet_distributuin_valid.txt', predit_score, fmt='%f', delimiter=',')
    # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_valid_fpr', fpr, fmt='%f', delimiter=',')
    # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_valid_tpr', tpr, fmt='%f', delimiter=',')

   
        
    validating_loss = valid_l_sum / n
    
    return validating_loss, valid_auc
    
def train_ch5(net, train_iter, valid_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    
    loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    
    train_loss=[]
    valid_loss=[]
    best_valid_auc = 0.0
    
    for epoch in range(num_epochs):
        
        n = 0.0
        train_l_sum = 0.0
        
        train_iter.reset()
        valid_iter.reset()
        
        
        predit_score = []
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
            predit_score.extend(y_hat.asnumpy().tolist())
        
        fpr, tpr, thresholds = metrics.roc_curve(target_label, predit_score, pos_label=1)
        train_auc = metrics.auc(fpr, tpr)
        
        training_loss = train_l_sum / n
        
        validating_loss, valid_auc = validating_auc(net, valid_iter, loss, ctx)
        
        
        train_loss.append(training_loss)
        valid_loss.append(validating_loss)
        
        # np.savetxt(os.getcwd() + '/202203/result_combined/TCN_mxnet_distributuin_train.txt', predit_score, fmt='%f', delimiter=',')

        
        
        print(epoch, training_loss, validating_loss, valid_auc)
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            
            if valid_auc > best_valid_auc:
                
                testing_loss, test_auc = testing_auc(net, test_iter, loss, ctx)
                
                
                
            # net.save_parameters(os.getcwd() + "/202203/params/TCN_mxnet_epoch" + str(epoch) + ".mxnet")
            # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_train_fpr', fpr, fmt='%f', delimiter=',')
            # np.savetxt(os.getcwd() + '/202203/result_combined/roc_TCN_mxnet_train_tpr', tpr, fmt='%f', delimiter=',')

    
    
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, 'b', x, valid_loss, 'r')
    print("The training_auc of the test_data test_data-hqz.npy:", train_auc, "The validing_auc:", valid_auc, "The testing_auc:", test_auc)
    
    return train_loss, valid_loss, testing_loss
    
batch_size = 64    
    
#%% input data

train_Data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
train_Label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')

train_data, valid_data, train_label, valid_label = train_test_split(train_Data, train_Label, stratify=train_Label, test_size=0.2, random_state=5)


data1_ = []
for i in range(len(train_data)):
    data1_.append(preprocessing.minmax_scale(np.array(train_data[i]).reshape(6,85),axis=1))
train_data = np.array(data1_).reshape(len(data1_), 1, 6, 85)

#test_data = np.load(os.getcwd() + '/data/Test_XICs.npy')
#test_label = np.load(os.getcwd() + '/data/Test_labels.npy')

test_data = np.load(os.getcwd() + '/data/test_data-hqz.npy')
test_label = np.load(os.getcwd() + '/data/test_label-hqz.npy')

data2_ = []
for i in range(len(valid_data)):
    data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6,85),axis=1))
    #data2_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6,85),axis=1))

valid_data = nd.array(data2_).reshape(len(data2_), 1, 6, 85)

data3_ = []
for i in range(len(test_data)):
    #data3_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6,85),axis=1))
    data3_.append(preprocessing.minmax_scale(np.array(test_data[i]).reshape(6,85),axis=1))

test_data = nd.array(data3_).reshape(len(data3_), 1, 6, 85)

data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True)
data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True)
data_iter_valid  = mx.io.NDArrayIter(valid_data, label=valid_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')


#%% 
ctx = mx.cpu()
lr, num_epochs = 0.001, 2


n_classes = 2
input_channels = 1
seq_length = int(6*85 / input_channels)
kernel_size = 3
dropout = 0.2
steps = 0


time_start = time.time()
net = Model(input_size=input_channels , output_size=n_classes, num_channels=1,
            kernel_size=kernel_size, dropout=dropout)
net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=mx.cpu())
net.cast('float32')
#optim = optimizer.Adam(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08,wd=1e-5)
trainer = gluon.Trainer(net.collect_params(), 'adam',optimizer_params={'learning_rate':lr,'wd':5e-4}) #{'wd':5e-4}
train_loss, valid_loss, testing_loss = train_ch5(net, data_iter_train, data_iter_valid, data_iter_test, batch_size, trainer, ctx, num_epochs)

time_end = time.time()
time_sum = time_end - time_start
print("TCN_mxnet time:", time_sum)



