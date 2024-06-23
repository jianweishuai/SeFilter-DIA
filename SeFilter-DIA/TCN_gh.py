import os
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from sklearn import preprocessing
import numpy as np
import argparse
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
#parser.add_argument('--cuda', action='store_false',
#                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
#if torch.cuda.is_available():
#   if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#%% from TCN.tcn import TemporalConvNet
import torch.nn as nn
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


#%% from TCN.mnist_pixel.utils import data_generator
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


batch_size = args.batch_size
n_classes = 2
input_channels = 1
seq_length = int(6*85 / input_channels)
epochs = args.epochs
steps = 0

print(args)

permute = torch.Tensor(np.random.permutation(6*85).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

#if args.cuda:
#    model.cuda()
#    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)





def train(ep):
    global steps
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        #if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            print('correct/100batch : {} / 6400 ({:.0f}%)\n'.format(correct, 
                  100. * correct / 6400))
            train_loss = 0
            correct = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #if args.cuda:
               # data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss




#%% Converting the input data to the proper stucture

train_Data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
train_Label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')

#train_data, valid_data, train_label, valid_label = train_test_split(train_Data, train_Label, stratify=train_Label, test_size=0.3, random_state=5)

#test_data = np.load(os.getcwd() + "/data/Test_XICs.npy")
#test_label = np.load(os.getcwd() + "/data/Test_labels.npy")

test_data = np.load(os.getcwd() + '/data/test_data-hqz.npy')
test_label = np.load(os.getcwd() + '/data/test_label-hqz.npy')

data1_ = []
for i in range(len(train_Data)):
	data1_.append(preprocessing.minmax_scale(np.array(train_Data[i]).reshape(6, 85), axis=1))
train_data = np.array(data1_).reshape(len(data1_), 1, 6*85)

#data2_ = []
#for i in range(len(valid_data)):
#	data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6, 85), axis=1))
#valid_data = np.array(data2_)

data3_ = []
for i in range(len(test_data)):
	#data3_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6, 85), axis=1))
    data3_.append(preprocessing.minmax_scale(np.array(test_data[i]).reshape(6, 85), axis=1))
test_data = np.array(data3_).reshape(len(data3_), 1, 6*85)

#print(train_data.shape)
#print(train_Label.sum())
#print(test_label.shape)

test_label = np.squeeze(test_label)
x_train = torch.from_numpy(train_data)
y_train = torch.from_numpy(train_Label)
x_test = torch.from_numpy(test_data)
y_test = torch.from_numpy(test_label)
x_train = x_train.float()
x_test = x_test.float()
y_train = y_train.long()
y_test = y_test.long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)


time_start = time.time()

for epoch in range(1, epochs+1):
    train(epoch)
    test()
    if epoch % 10 == 0:
        lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

time_end = time.time()

time_sum = time_end - time_start
print("TCN time:", time_sum)



















