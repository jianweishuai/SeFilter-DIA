# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:59:28 2022

@author: GuoHuan
"""
# import time
# import os
# import copy
# import random
# import numpy as np
# import mxnet as mx
# from mxnet import nd
# from mxnet.gluon import nn
# from mxnet import autograd
# from sklearn import metrics
# from mxnet import gluon, init
# from sklearn import preprocessing
# from mxnet.gluon import loss as gloss
# from sklearn.model_selection import train_test_split

from mxnet import np, npx, init ,autograd, gluon
from mxnet.gluon import rnn

import plotly.express as px
import pandas as pd
import math
npx.set_np()
ctx = npx.gpu() if npx.num_gpus() > 0 else npx.cpu()

batch_size, num_steps = 32, 35
train_iter, vocab = mxnet.load_data_time_machine(batch_size, num_steps)

#%%
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选存储单元
    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device), np.zeros((batch_size, num_hiddens), ctx=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)  # 输入门
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)  # 遗忘门
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)  # 输出门
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)  # 候选存储单元
        C = F * C + I * C_tilda   # 通过遗忘门控制之前的记忆单元输入 + 输入门控制候选存储输入
        H = O * np.tanh(C)  # 输出门控制 记忆单元输出为隐藏状态
        Y = np.dot(H, W_hq) + b_q  
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)

def predict(prefix, num_preds, model, vocab, device): 
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: np.array([outputs[-1]], ctx=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 开始预测
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(model, theta): 
    """裁剪梯度"""
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm



def train_epoch(model, train_iter, loss, updater, device, 
                    use_random_iter):
    state, start = None, time.time()
    l_sum, count = 0, 0  # 统计loss 和 count
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机采样时初始化state
            state = model.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = model(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # mean函数已经被调用
        l_sum += l * y.size
        count += y.size
    return math.exp(l_sum / count), count / (time.time() - start)


def train(model, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    epochs_lst, l_lst = [], []
    # Initialize
    if isinstance(model, gluon.Block):
        model.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    pre = lambda prefix: predict(prefix, 50, model, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(model, train_iter, loss, updater, device, use_random_iter)
        if epoch % 10 == 0:
            print(pre('time traveller'))
            epochs_lst.append(epoch+1)
            l_lst.append(ppl)
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(pre('time traveller'))
    print(pre('traveller'))
    fig = px.line(pd.DataFrame([epochs_lst, l_lst], index=['epoch', 'perplexity']).T, x='epoch', y='perplexity', 
            width=600, height=380)
    fig.show()


vocab_size, num_hiddens = len(vocab), 256
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, ctx, get_lstm_params,
                            init_lstm_state, lstm)
train(model, train_iter, vocab, lr, num_epochs, ctx)


# #%%
# # input data
# train_data = np.load(os.getcwd()+'/data/train_data_gh.npy')
# train_label = np.load(os.getcwd()+'/data/train_label_gh.npy')

# valid_data = np.load(os.getcwd()+'/data/valid_data_gh.npy')
# valid_label = np.load(os.getcwd()+'/data/valid_label_gh.npy')

# test_data  = np.load(os.getcwd()+'/data/test_data_gh.npy')
# test_label = np.load(os.getcwd()+'/data/test_label_gh.npy')

# train_data, train_label = nd.array(train_data).reshape(len(train_data),1, 6, 85), nd.array(train_label)   
# valid_data, valid_label = nd.array(valid_data).reshape(len(valid_data),1, 6, 85), nd.array(valid_label)   
# test_data, test_label = nd.array(test_data).reshape(len(test_data),1, 6, 85), nd.array(test_label)   


# data_iter_train = mx.io.NDArrayIter(train_data, label=train_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
# data_iter_test  = mx.io.NDArrayIter(test_data, label=test_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')
# data_iter_valid  = mx.io.NDArrayIter(valid_data, label=valid_label, batch_size=batch_size, shuffle=True, last_batch_handle='discard')


# #%%
# time_start = time.time()
# """======================== Train LSTM + CNN ========================"""







