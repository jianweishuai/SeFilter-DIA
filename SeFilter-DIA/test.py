import os
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
from sklearn import preprocessing
import numpy as np
import argparse
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset



#%% Converting the input data to the proper stucture

train_Data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
#train_Data = np.load(r'D:\C_files\gh\GraduateTime\data\nonshuffle_x.npy')
train_Label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')



#train_data, valid_data, train_label, valid_label = train_test_split(train_Data, train_Label, stratify=train_Label, test_size=0.3, random_state=5)

#test_data = np.load(os.getcwd() + "/data/Test_XICs.npy")
#test_label = np.load(os.getcwd() + "/data/Test_labels.npy")

test_data = np.load(os.getcwd() + '/data/test_data-hqz.npy')
test_label = np.load(os.getcwd() + '/data/test_label-hqz.npy')


data1_ = []
for i in range(len(train_Data)):
	data1_.append(preprocessing.minmax_scale(np.array(train_Data[i]).reshape(6, 85), axis=1))
train_data = np.array(data1_).reshape(len(data1_), 6, 85)

#data2_ = []
#for i in range(len(valid_data)):
#	data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6, 85), axis=1))
#valid_data = np.array(data2_)

data3_ = []
for i in range(len(test_data)):
	data3_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6, 85), axis=1))
test_data = np.array(data3_).reshape(len(data3_), 6, 85)

print('train_data_size:', train_data.shape)
print('train_label_positive:', train_Label.sum())
print('train_label_size:',train_Label.shape)
print('test_label_size:', test_label.shape)
print('train_label_positive:', test_label.sum())
print(test_label)
x_train = torch.from_numpy(train_data)
y_train = torch.from_numpy(train_Label)
x_test = torch.from_numpy(test_data)
y_test = torch.from_numpy(test_label)
x_train = x_train.float()
x_test = x_test.float()
y_train = y_train.long()
y_test = y_test.long()