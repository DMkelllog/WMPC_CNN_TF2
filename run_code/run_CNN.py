#!/usr/bin/env python
# coding: utf-8


import pickle
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.densenet import DenseNet121


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


REP_ID = int(sys.argv[1])
TRAIN_SIZE_ID = int(sys.argv[2])
MODEL_ID = int(sys.argv[3])

BATCH_SIZE = 32
MAX_EPOCH = 1000
TRAIN_SIZE_LIST = [500, 5000, 50000, 162946]
lr = 1e-4
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

with open('../data/X_CNN_64.pickle', 'rb') as f:
    X_resize = pickle.load(f)
with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f)

X_resize = (X_resize - 0.5) * 2
X_resize_stacked = np.repeat(X_resize, 3, -1)
y_onehot = tf.keras.utils.to_categorical(y)


RAN_NUM = 27407+REP_ID
    

TRAIN_SIZE = TRAIN_SIZE_LIST[TRAIN_SIZE_ID]
        
#Model ID = 0: VGG, 1: ResNEt, 2: DenseNet
X_trnval, X_tst, y_trnval, y_tst =  train_test_split(X_resize_stacked, y_onehot, 
                                                     test_size=10000, random_state=RAN_NUM)

# Randomly sample train set for evaluation at various training set size
if TRAIN_SIZE == X_trnval.shape[0]:
    pass
else:
    X_trnval,_ , y_trnval, _ = train_test_split(X_trnval, y_trnval, 
                                                train_size=TRAIN_SIZE, random_state=RAN_NUM)
# Get unique labels in training set. Some labels might not appear in small training set.
labels = np.unique(np.argmax(y_trnval, 1))

if MODEL_ID == 0: base_model = VGG16(weights=None, pooling='avg', include_top=False)
elif MODEL_ID == 1: base_model = ResNet50V2(weights=None, pooling='avg', include_top=False)
elif MODEL_ID == 2: base_model = DenseNet121(weights=None, pooling='avg', include_top=False)
predictions = tf.keras.layers.Dense(9, activation='softmax')(base_model.output)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers: layer.trainable = True
model.compile(optimizer= tf.keras.optimizers.Adam(lr=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

log = model.fit(X_trnval, y_trnval, validation_split=0.2, 
                batch_size=BATCH_SIZE, epochs=MAX_EPOCH, 
                callbacks=[early_stopping], verbose=0)
y_trnval_hat= model.predict(X_trnval)
y_tst_hat= model.predict(X_tst)

macro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='macro')
micro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='micro')
cm = confusion_matrix(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1))

filename = '../result/CNN/WMPC_'+'CNN_'+str(MODEL_ID)+'_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'

with open(filename+'softmax.pickle', 'wb') as f:
    pickle.dump([y_trnval_hat, y_tst_hat], f)
with open(filename+'f1_score.pickle', 'wb') as f:
    pickle.dump([macro, micro, cm], f)

print('model_id:', MODEL_ID,
      'train size:', TRAIN_SIZE,
      'rep_id:', REP_ID,
      'macro:', np.round(macro, 4), 
      'micro:', np.round(micro, 4))

