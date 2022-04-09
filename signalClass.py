########################################
#Created by Nicholas Rasmussen on 4-9-22
#USD Computer Science Undergrad
########################################

import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.constraints import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
K.set_learning_phase(1)
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc
from scipy import interp
from audiomentations import *
import matplotlib.pyplot as plt
from itertools import cycle
import ast
import time
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import random
import copy
from sklearn.linear_model import SGDClassifier

#audiomentation's augmentation to modify upsampled data
augment1 = Compose([
    #Shift(min_fraction=-.05, max_fraction=.05, p=1),
    AddGaussianSNR(min_SNR=.001, max_SNR=.01, p=1)
])

#importing data
data = pd.read_csv("./raw_er.csv", header = None)

#Variable to hold entire dataset and loop to iterate over dataframe
dataset = []
for row in data.itertuples():
    #variable for a zeros array to hold values
    values = np.zeros(1877)
    count = 0
    #loop to iterate over all values in the samples excluding the final values which was for classification
    for x in row[:-1]:
        #eliminates all noise below the values of 1
        if (x < 1):
            values[count] = float(0)
            count+=1
            continue
        #sqrt value to reduce all noise above 1 
        values[count] = float(x)**.5
        count+=1


    # appending array to the dataset list with the class
    dataset.append((values,0))

#creating variable to hold limit to balance the dataset with undersampling
limit = len(dataset)

#import data and loop over dataframe for the other class
data2 = pd.read_csv("./raw_nr.csv", header = None)
count2 = 0
for row in data2.itertuples():
    #if (count2>=limit):break
    values = np.zeros(1877)
    count1 = 0
    
    
    for x in row[:-1]:
        if (x < 1):
            values[count1] = float(0)
            count1+=1
            continue
        values[count1] = float(x)**.5
        count1+=1
    
    count2+=1
    dataset.append((values,1))

#shuffling dataset
random.shuffle(dataset)
random.shuffle(dataset)
random.shuffle(dataset)

#Splitting dataset into train and validation, eventually I need to modify the code to ensure the same exact ratio of pos/negs is in each split. 
#However, for a protoype it doesn't matter a bunch, it may even add a degree of robustness showing that the model is resistant to imbalances.
train = dataset[:int(.8*len(dataset))]
val = dataset[int(.8*len(dataset)):]

#variable to hold augmented, approximately balanced, training set
train2 = train.copy()

#loop to creat needed augmented samples
for (x,y) in train:
    #negative class augments
    if (y == 0):
        #creating two augmented samples of the negative Ele class to achieve an approximately balanced dataset
        for i in range(2):
            z = augment1(x,1)
            train2.append((z,0))
        
    #positive class augments
    if (y == 1):
        for i in range(0):
            z = augment1(x,1)
            train2.append((z,1))
        
'''
#you can use this code for direct upsampling
for (x,y) in train:
    
    if (y == 0):
        train2.append((x,0))
        train2.append((x,0))
'''


#shuffle
random.shuffle(train2)
random.shuffle(train2)
random.shuffle(train2)

#creating variables to separate data from labels
X_test, y_test = zip(*val)
X_train, y_train = zip(*train2)


#this scaler can be used for normailzed data, it does not perform well on the raw data
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

#figuring out how many approximate N nieghbores to use
import math
print(math.sqrt(len(y_test)))

#creating a KNN classifier. KNN does not perform well on raw data since there is noise between -10/10.
classifier = KNeighborsClassifier(n_neighbors=17,p=2,metric='minkowski',weights='uniform', algorithm='brute', leaf_size=8, metric_params=None,)
classifier.fit(X_train,y_train)

#use classifyer to predict the test set
y_pred =  classifier.predict(X_test)

#confusing matrix for certain metrics
cm= confusion_matrix(y_test,y_pred)

#print(f1_score(y_test,y_pred))

#print(accuracy_score(y_test,y_pred))

#classification report has all the metric you really need.
print(classification_report(y_test,y_pred))

from sklearn.svm import SVC

# kernel to be set linear as it is binary class - Sample proceedure as KNN
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

y_pred =  classifier.predict(X_test)

cm= confusion_matrix(y_test,y_pred)

#print(f1_score(y_test,y_pred))

#print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))
'''
#you can use the code below to create a nural network if you wish, i haven't had much luck with it.
#init = tensorflow.keras.initializers.HeNormal(seed=1)
#reg = regularizers.l1(l1=0.01)
reg = None
model = Sequential()
input_shape=(1877,)

#model.add(Bidirectional(GRU(64, return_sequences=True, kernel_regularizer= reg,recurrent_activation="sigmoid")))
#model.add(BatchNormalization())
#model.add(Dropout(rate=0.2))

#model.add(Bidirectional(GRU(32, return_sequences=True, kernel_regularizer= reg,recurrent_activation="sigmoid")))
#model.add(BatchNormalization())
#model.add(Dropout(rate=0.2))

model.add(Dense(64,input_shape=input_shape, kernel_regularizer = reg))
model.add(Dropout(rate=0.2))
model.add(Activation('relu'))
model.add(LeakyReLU())

#model.add(GlobalMaxPooling1D())

model.add(Dense(32,kernel_regularizer = reg))
model.add(Dropout(rate=0.2))
model.add(Activation('relu'))
model.add(LeakyReLU())

model.add(Dense(1,kernel_regularizer = reg))
model.add(Activation('softmax'))

#modifing the data for to be inserted into nueral network
X_train = np.array([x.reshape( (1877, ) ) for x in X_train])
X_test = np.array([x.reshape( (1877, ) ) for x in X_test])
y_train = np.array(y_train)
y_test = np.array(y_test)

model.compile(optimizer = RMSprop(lr = .0001), loss = 'binary_crossentropy', metrics = ['accuracy', 'Precision', 'AUC', tensorflow.keras.metrics.Recall(), tensorflow.keras.metrics.TrueNegatives()])

count = 0

model_checkpoint = [
        ModelCheckpoint(filepath = './Checks/check' + str(count) + ".hdf5", monitor='val_auc',verbose=1, save_best_only=True, mode = 'max'),
        #tensorflow.keras.callbacks.EarlyStopping(monitor="val_auc", min_delta=0, patience=10, verbose=0, mode="max", baseline=None, restore_best_weights=True)
    ]

model.fit(
        x=X_train,
        y=y_train,
        epochs=100,
        batch_size=128,
        validation_data= (X_test, y_test),
        callbacks=[model_checkpoint],
        class_weight={0: 1, 1: 1},
        shuffle = True)

model.load_weights('./Checks/check' + str(count) + '.hdf5')

score = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size = 1)
'''