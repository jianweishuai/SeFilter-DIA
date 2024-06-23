# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:09:34 2022

@author: GuoHuan
"""
import time
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
from sklearn import preprocessing
from mxnet.gluon import loss as gloss
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#%%
# input data

def gh_shuffle(data, label):
    
    result = list(zip(data, label))  
    # 打乱的索引序列
    np.random.shuffle(result)
    data, label = zip(*result)
    return data, label
'''
train_data = np.load(os.getcwd()+'/data/train_data_gh.npy')
train_label = np.load(os.getcwd()+'/data/train_label_gh.npy')

valid_data = np.load(os.getcwd()+'/data/valid_data_gh.npy')
valid_label = np.load(os.getcwd()+'/data/valid_label_gh.npy')

test_data  = np.load(os.getcwd()+'/data/test_data_gh.npy')
test_label = np.load(os.getcwd()+'/data/test_label_gh.npy')
'''
time_input = time.time()

train_Data = np.load(os.getcwd() + '/data/nonshuflle_x.npy')
train_Label = np.load(os.getcwd() + '/data/nonshuffle_label.npy')

train_data, valid_data, train_label, valid_label = train_test_split(train_Data, train_Label, stratify=train_Label, test_size=0.2, random_state=5)

test_data = np.load(os.getcwd() + '/data/test_data-hqz.npy')
test_label = np.load(os.getcwd() + '/data/test_label-hqz.npy')
#test_data = np.load(os.getcwd() + '/data/Test_XICs.npy')
#test_label = np.load(os.getcwd() + '/data/Test_labels.npy')

# Normalization
data1_ = []
for i in range(len(train_data)):
    data1_.append(preprocessing.minmax_scale(np.array(train_data[i]).reshape(6,85),axis=1))
train_data = np.array(data1_)

data2_ = []
for i in range(len(valid_data)):
    data2_.append(preprocessing.minmax_scale(np.array(valid_data[i]).reshape(6,85),axis=1))
    #data2_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6,85),axis=1))
valid_data = np.array(data2_)

data3_ = []
for i in range(len(test_data)):
    data3_.append(preprocessing.minmax_scale(np.array(test_data[i]).reshape(6,85),axis=1))
    #data2_.append(preprocessing.minmax_scale(np.array(test_data[i][1:]).reshape(6,85),axis=1))
test_data = np.array(data3_)

train_data = train_data.reshape(len(train_data), 6*85)
valid_data = valid_data.reshape(len(valid_data), 6*85)
test_data = test_data.reshape(len(test_data), 6*85)

# shuffle
train_data, train_label = gh_shuffle(train_data, train_label)
valid_data, valid_label = gh_shuffle(valid_data, valid_label)
test_data, test_label = gh_shuffle(test_data, test_label)

# print(train_label)
# print(test_label[0:300])


time_input_end = time.time()
time_sum = time_input_end - time_input
print('time of input data', time_sum)

time_end = time.time()

'''
#%%
"""======================== Train LDA ========================"""

clf_LDA = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=None, n_components=1, store_covariance=False, tol=0.0001)
clf_LDA_model = clf_LDA.fit(train_data, train_label)
 
train_predict_LDA_label = clf_LDA_model.predict(train_data)  

fpr_train_LDA, tpr_train_LDA , thresholds = metrics.roc_curve(train_label, train_predict_LDA_label, pos_label=1)

train_auc = metrics.auc(fpr_train_LDA, tpr_train_LDA)

valid_predict_LDA_label = clf_LDA_model.predict(valid_data)

fpr_valid_LDA, tpr_valid_LDA , thresholds = metrics.roc_curve(valid_label, valid_predict_LDA_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_LDA, tpr_valid_LDA)

test_predict_LDA_label = clf_LDA_model.predict(test_data)

fpr_test_LDA, tpr_test_LDA , thresholds = metrics.roc_curve(test_label, test_predict_LDA_label, pos_label=1)

test_auc = metrics.auc(fpr_test_LDA, tpr_test_LDA)
print('一、 LDA: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_LDA = time_end - time_before
print('time_LDA: ', time_LDA)


#%%
"""======================== Train QDA ========================"""
clf_QDA = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
clf_QDA_model = clf_QDA.fit(train_data, train_label)

train_predict_QDA_label = clf_QDA_model.predict(train_data)  

fpr_train_QDA, tpr_train_QDA , thresholds = metrics.roc_curve(train_label, train_predict_QDA_label, pos_label=1)

train_auc = metrics.auc(fpr_train_QDA, tpr_train_QDA)

valid_predict_QDA_label = clf_QDA_model.predict(valid_data)

fpr_valid_QDA, tpr_valid_QDA , thresholds = metrics.roc_curve(valid_label, valid_predict_QDA_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_QDA, tpr_valid_QDA)

test_predict_QDA_label = clf_QDA_model.predict(test_data)

fpr_test_QDA, tpr_test_QDA , thresholds = metrics.roc_curve(test_label, test_predict_QDA_label, pos_label=1)

test_auc = metrics.auc(fpr_test_QDA, tpr_test_QDA)
print('二、 QDA: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_QDA = time_end - time_before
print('time_QDA: ', time_QDA)

'''
#%%
"""======================== Train KNN Classifier  ========================"""
from sklearn.model_selection import GridSearchCV

param_grid = {"leaf_size":[20, 30],
              "p":[1, 2, 4],
              'weights': ['uniform', 'distance'],
              'n_neighbors': [1, 2]}
print("Parameters:{}".format(param_grid))

grid_search = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5) #实例化一个GridSearchCV类，第一个为模型，第二个为超参数值，第三个为交叉验证个数

clf_KNC_model = grid_search.fit(train_data, train_label) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。


# clf_KNC = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=10, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# clf_KNC_model = clf_KNC.fit(train_data, train_label)

train_predict_KNC_label = clf_KNC_model.predict(train_data)

fpr_train_KNC, tpr_train_KNC , thresholds = metrics.roc_curve(train_label, train_predict_KNC_label, pos_label=1)

train_auc = metrics.auc(fpr_train_KNC, tpr_train_KNC)

valid_predict_KNC_label = clf_KNC_model.predict(valid_data)

fpr_valid_KNC, tpr_valid_KNC , thresholds = metrics.roc_curve(valid_label, valid_predict_KNC_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_KNC, tpr_valid_KNC)

test_predict_KNC_label = clf_KNC_model.predict(test_data)

fpr_test_KNC, tpr_test_KNC , thresholds = metrics.roc_curve(test_label, test_predict_KNC_label, pos_label=1)

test_auc = metrics.auc(fpr_test_KNC, tpr_test_KNC)
print('三、 KNC: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_KNC = time_end - time_before
print('time_KNC: ', time_KNC)


'''
#%%
"""======================== Train LogisticRegression ========================"""

clf_LR = LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, 
                            fit_intercept=True, intercept_scaling=1, class_weight=None,
                            random_state=None, solver='lbfgs', max_iter=100, 
                            multi_class='auto', verbose=0, warm_start=False, 
                            n_jobs=None, l1_ratio=None)
clf_LR_model = clf_LR.fit(train_data, train_label)

train_predict_LR_label = clf_LR_model.predict(train_data)
fpr_train_LR, tpr_train_LR , thresholds = metrics.roc_curve(train_label, train_predict_LR_label, pos_label=1)

train_auc = metrics.auc(fpr_train_LR, tpr_train_LR)

valid_predict_LR_label = clf_LR_model.predict(valid_data)

fpr_valid_LR, tpr_valid_LR , thresholds = metrics.roc_curve(valid_label, valid_predict_LR_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_LR, tpr_valid_LR)

test_predict_LR_label = clf_LR_model.predict(test_data)

fpr_test_LR, tpr_test_LR , thresholds = metrics.roc_curve(test_label, test_predict_LR_label, pos_label=1)

test_auc = metrics.auc(fpr_test_LR, tpr_test_LR)
print('四、 LR: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_LR = time_end - time_before
print('time_LR: ', time_LR)


#%%
"""======================== Train RandomForestClassifier ========================"""
 
clf_RandForest = RandomForestClassifier()
clf_RandForest_model = clf_RandForest.fit(train_data, train_label)

train_predict_RandForest_label = clf_RandForest_model.predict(train_data)

fpr_train_RandForest, tpr_train_RandForest , thresholds = metrics.roc_curve(train_label, train_predict_RandForest_label, pos_label=1)

train_auc = metrics.auc(fpr_train_RandForest, tpr_train_RandForest)

valid_predict_RandForest_label = clf_RandForest_model.predict(valid_data)

fpr_valid_RandForest, tpr_valid_RandForest , thresholds = metrics.roc_curve(valid_label, valid_predict_RandForest_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_RandForest, tpr_valid_RandForest)

test_predict_RandForest_label = clf_RandForest_model.predict(test_data)

fpr_test_RandForest, tpr_test_RandForest , thresholds = metrics.roc_curve(test_label, test_predict_RandForest_label, pos_label=1)

test_auc = metrics.auc(fpr_test_RandForest, tpr_test_RandForest)
print('五、 RandForest: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_RandForest = time_end - time_before
print('time_RandForest: ', time_RandForest)

#%%
"""======================== Train Tree ========================"""

clf_Tree = tree.DecisionTreeClassifier()
clf_Tree_model = clf_Tree.fit(train_data, train_label)

train_predict_Tree_label = clf_Tree_model.predict(train_data)

fpr_train_Tree, tpr_train_Tree , thresholds = metrics.roc_curve(train_label, train_predict_Tree_label, pos_label=1)

train_auc = metrics.auc(fpr_train_Tree, tpr_train_Tree)

valid_predict_Tree_label = clf_Tree_model.predict(valid_data)

fpr_valid_Tree, tpr_valid_Tree , thresholds = metrics.roc_curve(valid_label, valid_predict_Tree_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_Tree, tpr_valid_Tree)

test_predict_Tree_label = clf_Tree_model.predict(test_data)

fpr_test_Tree, tpr_test_Tree , thresholds = metrics.roc_curve(test_label, test_predict_Tree_label, pos_label=1)

test_auc = metrics.auc(fpr_test_Tree, tpr_test_Tree)
print('六、 Tree: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_Tree = time_end - time_before
print('time_Tree: ', time_Tree)



#%%
"""======================== Train GradientBoostingClassifier ========================"""
 
clf_GBC = GradientBoostingClassifier(n_estimators=200)
clf_GBC_model = clf_GBC.fit(train_data, train_label)

train_predict_GBC_label = clf_GBC_model.predict(train_data)

fpr_train_GBC, tpr_train_GBC , thresholds = metrics.roc_curve(train_label, train_predict_GBC_label, pos_label=1)

train_auc = metrics.auc(fpr_train_GBC, tpr_train_GBC)

valid_predict_GBC_label = clf_GBC_model.predict(valid_data)

fpr_valid_GBC, tpr_valid_GBC , thresholds = metrics.roc_curve(valid_label, valid_predict_GBC_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_GBC, tpr_valid_GBC)

test_predict_GBC_label = clf_GBC_model.predict(test_data)

fpr_test_GBC, tpr_test_GBC , thresholds = metrics.roc_curve(test_label, test_predict_GBC_label, pos_label=1)

test_auc = metrics.auc(fpr_test_GBC, tpr_test_GBC)
print('七、 GBC: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_GBC = time_end - time_before
print('time_GBC: ', time_GBC)

#%%
"""======================== Train AdaBoostClassifier ========================"""

clf_AdaBC = AdaBoostClassifier()
clf_AdaBC_model = clf_AdaBC.fit(train_data, train_label)

train_predict_AdaBC_label = clf_AdaBC_model.predict(train_data)

fpr_train_AdaBC, tpr_train_AdaBC , thresholds = metrics.roc_curve(train_label, train_predict_AdaBC_label, pos_label=1)

train_auc = metrics.auc(fpr_train_AdaBC, tpr_train_AdaBC)

valid_predict_AdaBC_label = clf_AdaBC_model.predict(valid_data)

fpr_valid_AdaBC, tpr_valid_AdaBC , thresholds = metrics.roc_curve(valid_label, valid_predict_AdaBC_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_AdaBC, tpr_valid_AdaBC)

test_predict_AdaBC_label = clf_AdaBC_model.predict(test_data)

fpr_test_AdaBC, tpr_test_AdaBC , thresholds = metrics.roc_curve(test_label, test_predict_AdaBC_label, pos_label=1)

test_auc = metrics.auc(fpr_test_AdaBC, tpr_test_AdaBC)
print('八、 AdaBC: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_AdaBC = time_end - time_before
print('time_AdaBC: ', time_AdaBC)


#%%
"""======================== Train GaussianNB ========================"""

clf_GaussianNB = GaussianNB()
clf_GaussianNB_model = clf_GaussianNB.fit(train_data, train_label)

train_predict_GaussianNB_label = clf_GaussianNB_model.predict(train_data)

fpr_train_GaussianNB, tpr_train_GaussianNB , thresholds = metrics.roc_curve(train_label, train_predict_GaussianNB_label, pos_label=1)

train_auc = metrics.auc(fpr_train_GaussianNB, tpr_train_GaussianNB)

valid_predict_GaussianNB_label = clf_GaussianNB_model.predict(valid_data)

fpr_valid_GaussianNB, tpr_valid_GaussianNB , thresholds = metrics.roc_curve(valid_label, valid_predict_GaussianNB_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_GaussianNB, tpr_valid_GaussianNB)

test_predict_GaussianNB_label = clf_GaussianNB_model.predict(test_data)

fpr_test_GaussianNB, tpr_test_GaussianNB , thresholds = metrics.roc_curve(test_label, test_predict_GaussianNB_label, pos_label=1)

test_auc = metrics.auc(fpr_test_GaussianNB, tpr_test_GaussianNB)
print('九、 GaussianNB: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_GaussianNB = time_end - time_before
print('time_GaussianNB: ', time_GaussianNB)


#%%
"""======================== Train SVM ========================"""
 
clf_SVM = SVC(kernel='rbf', probability=True)
clf_SVM_model = clf_SVM.fit(train_data, train_label)

train_predict_SVM_label = clf_SVM_model.predict(train_data)

fpr_train_SVM, tpr_train_SVM , thresholds = metrics.roc_curve(train_label, train_predict_SVM_label, pos_label=1)

train_auc = metrics.auc(fpr_train_SVM, tpr_train_SVM)

valid_predict_SVM_label = clf_SVM_model.predict(valid_data)

fpr_valid_SVM, tpr_valid_SVM , thresholds = metrics.roc_curve(valid_label, valid_predict_SVM_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_SVM, tpr_valid_SVM)

test_predict_SVM_label = clf_SVM_model.predict(test_data)

fpr_test_SVM, tpr_test_SVM , thresholds = metrics.roc_curve(test_label, test_predict_SVM_label, pos_label=1)

test_auc = metrics.auc(fpr_test_SVM, tpr_test_SVM)
print('十、 SVM: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_SVM = time_end - time_before
print('time_SVM: ', time_SVM)


#%%
"""======================== Train MultinomialNB ========================"""
 
clf_MNB = MultinomialNB(alpha=0.01)
clf_MNB_model = clf_MNB.fit(train_data, train_label)

train_predict_MNB_label = clf_MNB_model.predict(train_data)

fpr_train_MNB, tpr_train_MNB , thresholds = metrics.roc_curve(train_label, train_predict_MNB_label, pos_label=1)

train_auc = metrics.auc(fpr_train_MNB, tpr_train_MNB)

valid_predict_MNB_label = clf_MNB_model.predict(valid_data)

fpr_valid_MNB, tpr_valid_MNB , thresholds = metrics.roc_curve(valid_label, valid_predict_MNB_label, pos_label=1)

valid_auc = metrics.auc(fpr_valid_MNB, tpr_valid_MNB)

test_predict_MNB_label = clf_MNB_model.predict(test_data)

fpr_test_MNB, tpr_test_MNB , thresholds = metrics.roc_curve(test_label, test_predict_MNB_label, pos_label=1)

test_auc = metrics.auc(fpr_test_MNB, tpr_test_MNB)
print('十一、 MNB: test_auc',test_auc,'train_auc',train_auc,'valid_auc',valid_auc)

time_before = time_end
time_end = time.time() 
time_MNB = time_end - time_before
print('time_MNB: ', time_MNB)
'''
