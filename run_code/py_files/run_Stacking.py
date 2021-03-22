#!/usr/bin/env python
# coding: utf-8


import pickle
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f)
y_onehot = tf.keras.utils.to_categorical(y)

REP_ID = int(sys.argv[1])
TRAIN_SIZE_ID = int(sys.argv[2])
MODEL_ID = int(sys.argv[3])

BATCH_SIZE = 32
MAX_EPOCH = 1000
TRAIN_SIZE_LIST = [500, 5000, 50000, 162946]
lr = 1e-4
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
MODEL_ID_CNN = 0
MODEL_ID_MFE = 4


def FNN(lr=1e-4):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(9, activation='softmax')])
    model.compile(optimizer= tf.keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


RAN_NUM = 27407+REP_ID
    

TRAIN_SIZE = TRAIN_SIZE_LIST[TRAIN_SIZE_ID]
        

y_trnval, y_tst =  train_test_split(y_onehot, test_size=10000, random_state=RAN_NUM)
if TRAIN_SIZE == 162946:
        pass
else:    
    y_trnval, _ = train_test_split(y_trnval, train_size=TRAIN_SIZE, random_state=RAN_NUM)

filename_MFE = '../result/MFE/WMPC_'+'MFE_'+str(MODEL_ID_MFE)+'_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'
filename_CNN = '../result/CNN/WMPC_'+'CNN_'+str(MODEL_ID_CNN)+'_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'

with open(filename_MFE + 'softmax.pickle', 'rb') as f:
    y_trnval_hat_mfe, y_tst_hat_mfe = pickle.load(f)
with open(filename_CNN + 'softmax.pickle', 'rb') as f:
    y_trnval_hat_cnn, y_tst_hat_cnn = pickle.load(f)
X_trnval_hat_concat = np.concatenate([y_trnval_hat_mfe, y_trnval_hat_cnn], axis=1)
X_tst_hat_concat = np.concatenate([y_tst_hat_mfe, y_tst_hat_cnn], axis=1)
labels = list(set(np.argmax(y_trnval, 1)))

if MODEL_ID == 0: 
    model = Ridge(alpha=0.1)
    model.fit(X_trnval_hat_concat, y_trnval)
elif MODEL_ID == 1: 
    model = DecisionTreeClassifier()
    model.fit(X_trnval_hat_concat, y_trnval)
elif MODEL_ID == 2:
    model = FNN()
    log = model.fit(X_trnval_hat_concat, y_trnval, validation_split=0.2, 
                    epochs=MAX_EPOCH, batch_size=BATCH_SIZE,
                    callbacks=[early_stopping], verbose=0)

y_tst_hat = model.predict(X_tst_hat_concat)
macro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='macro')
micro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='micro')
cm = confusion_matrix(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1))

filename = '../result/Stacking/WMPC_'+'Stacking_'+str(MODEL_ID)+'_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'
with open(filename+'f1_score.pickle', 'wb') as f:
    pickle.dump([macro, micro, cm], f)
if MODEL_ID == 0:
    with open(filename+'coef_.pickle', 'wb') as f:
        pickle.dump(model.coef_, f)

