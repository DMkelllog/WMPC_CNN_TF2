#!/usr/bin/env python
# coding: utf-8


import pickle
import os
import sys

from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

with open('../data/X_MFE.pickle', 'rb') as f:
    X_mfe = pickle.load(f)
with open('../data/X_CNN_64.pickle', 'rb') as f:
    X_cnn = pickle.load(f)
with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f)
    
y_onehot = tf.keras.utils.to_categorical(y)
X_cnn = (X_cnn-0.5)*2
X_cnn_stacked = np.repeat(X_cnn, 3, -1)

REPLICATION = 10
BATCH_SIZE = 32
MAX_EPOCH = 1000
TRAIN_SIZE_LIST = [500, 5000, 50000, 162946]
early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0)


REP_ID = int(sys.argv[1])
TRAIN_SIZE_ID = int(sys.argv[2])
MODEL_ID = int(sys.argv[3])


def FNN(lr=1e-4):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(9, activation='softmax')])
    model.compile(optimizer= tf.keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_multiNN():
    mfe_input = Input(shape=(59,))
    base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    concat = Concatenate()([mfe_input, base_model.output])
    dense_1 = Dense(128, activation='relu')(concat)
    dense_2 = Dense(128, activation='relu')(dense_1)
    prediction = Dense(9, activation='softmax')(dense_2)
    
    encoder = tf.keras.Model(inputs=[mfe_input, base_model.input], outputs=concat)
    model = tf.keras.Model(inputs=[mfe_input, base_model.input], outputs=prediction)
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics='accuracy')
    return model

RAN_NUM = 27407+REP_ID
    

TRAIN_SIZE = TRAIN_SIZE_LIST[TRAIN_SIZE_ID]
        
X_trnval_cnn, X_tst_cnn, y_trnval, y_tst =  train_test_split(X_cnn_stacked, y_onehot, 
                                                             test_size=10000, random_state=RAN_NUM)
X_trnval_mfe, X_tst_mfe =  train_test_split(X_mfe, 
                                                     test_size=10000, random_state=RAN_NUM)
# Randomly sample train set for evaluation at various train set size
if TRAIN_SIZE == X_trnval_mfe.shape[0]:
    pass
else:
    X_trnval_cnn, _, y_trnval, _ = train_test_split(X_trnval_cnn, y_trnval, 
                                                train_size=TRAIN_SIZE, random_state=RAN_NUM)
    X_trnval_mfe, _, = train_test_split(X_trnval_mfe, 
                                                train_size=TRAIN_SIZE, random_state=RAN_NUM)

labels = np.unique(np.argmax(y_trnval, 1))

scaler = StandardScaler()
X_trnval_mfe_scaled = scaler.fit_transform(X_trnval_mfe)
X_tst_mfe_scaled = scaler.transform(X_tst_mfe)

model = build_multiNN()


log = model.fit([X_trnval_mfe_scaled, X_trnval_cnn], y_trnval, validation_split=0.2, 
              epochs=MAX_EPOCH, callbacks=[early_stopping], verbose=0)
y_trnval_hat= model.predict([X_trnval_mfe_scaled, X_trnval_cnn])
y_tst_hat= model.predict([X_tst_mfe_scaled, X_tst_cnn])

macro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='macro')
micro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='micro')
cm = confusion_matrix(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1))

filename = '../result/MultiNN/WMPC_MultiNN_'+str(MODEL_ID)+'_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'

with open(filename+'f1_score.pickle', 'wb') as f:
    pickle.dump([macro, micro, cm], f)

print('model_id:', MODEL_ID,
      'train size:', TRAIN_SIZE,
      'rep_id:', REP_ID,
      'macro:', np.round(macro, 4), 
      'micro:', np.round(micro, 4),
      'labels:', len(labels),
      'epochs:', len(log.history['loss']))
