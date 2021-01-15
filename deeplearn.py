#!/usr/bin/env python

## the deep learning
## note book for genoAI

# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traingen import prep_trainset
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, activations, models, metrics
from tensorflow.keras import losses, optimizers, callbacks, utils

# %% USER SETTINGS
DATA_PKL = "train_data.p"

# %% FUNCTIONS
def loss_acc_plots(history):
    '''
    Generate 
    '''
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs   = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def z_norm(x_arr):
    '''
    compute column-level z-score
    '''
    x     = x_arr - np.mean(x_arr, axis = 0)
    y_arr = x/np.std(x_arr, axis = 0)

    return y_arr

# %% GET DATA
data_dct = prep_trainset(DATA_PKL)
data_dct.keys()

# %% PROCESS
data_exp    = data_dct['data_exp']
labs_enc    = data_dct['labels']

lb = LabelBinarizer()
ohe = OneHotEncoder()
# labs_trf = lb.fit_transform(labs_enc)
labs_trf = ohe.fit_transform(labs_enc.reshape(-1,1)).toarray()

# transform
data_exp    = z_norm(data_exp)

# %% PAD
data_exp_pad = np.pad(data_exp, ((0, 0), (0, 1)), mode='constant')
print(f"Original shape:{data_exp.shape} | New Shape:{data_exp_pad.shape}")

# %% RESHAPE
dexp_4d = data_exp_pad.reshape(data_exp_pad.shape[:-1] + (12,8,1)) ## For 2D CNN
dexp_3d = np.expand_dims(data_exp, axis = 2) ## For 1D CNN
dexp_3d.shape

# %% SPLITS
dexp = dexp_3d
d_trn, d_tst, l_trn, l_tst = train_test_split(dexp, labs_trf, test_size=0.1, stratify = labs_trf) 

# %% MODELS
def simple_cnn():
    '''
    Simple (seqeunntial) NN for testing
    '''

    l0 = layers.InputLayer((12,8,1), name = "input")

    a1 = layers.Conv2D(16, (2,2), activation=activations.relu, name = "c2d_a1")
    a2 = layers.Conv2D(16, (2,2), activation=activations.relu, name = "c2d_a2")
    a3 = layers.MaxPool2D((2,2), name = "pool_a3")
    a4 = layers.Dropout(0.25, name = "drop_1")

    b1 = layers.Conv2D(32, (2,2), activation=activations.relu, name = "c2d_b1")
    b2 = layers.MaxPool2D((2,2), name = "pool_b2")
    b3 = layers.Dropout(0.25, name = "drop_2")

    l1 = layers.Flatten(name = "flat")
    # l2 = layers.Dropout(0.20, name = "drop_3")

    l3 = layers.Dense(32, activation=activations.swish)
    l4 = layers.Dense(16, activation=activations.swish)
    lf = layers.Dense(1,  activation=activations.sigmoid)

    model = models.Sequential([l0, a1, a2, a3, a4, b1, b2, b3, l1, l3, l4, lf])

    return model

def simple_1D_cnn(data, labs):
    '''
    Very simple 1D CNN

    Shape: https://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape/43236878
    '''
    i,j,k = data.shape
    print(f"Shape:{i,j,k}")

    # input = layers.InputLayer(input_shape = (95,1), name ="input") ## this didn't work
    input = tf.keras.Input(shape=(95,1))

    x = layers.Conv1D(16, 3, input_shape = (95,1), activation=activations.relu, name = "c2d_a1")(input)
    x = layers.MaxPool1D(2, name  = "pool_a2")(x)
    x = layers.Dropout(0.25, name = "drop_1")(x)

    x = layers.Conv1D(16, 3, activation=activations.relu, name = "c2d_b1")(x)
    x = layers.MaxPool1D(2, name  = "pool_b2")(x)
    x = layers.Dropout(0.25, name = "drop_2")(x)

    d = layers.Flatten(name       = "flat")(x)

    d = layers.Dense(32, activation=activations.swish)(d)
    d = layers.Dense(16, activation=activations.swish)(d)
    output = layers.Dense(2,  activation=activations.softmax)(d)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

# %% SUMMARY
## summary
# model = simple_cnn()
model = simple_1D_cnn(d_trn, l_trn)
model.summary()

## plot
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi= 64)


# %% COMPILE
model.compile(optimizer ='rmsprop',
              loss      = losses.categorical_crossentropy,
              metrics   = ['accuracy']) ## metrics.RecallAtPrecision(precision=0.5)

# %% Fit
history = model.fit(x = d_trn,
                    y = l_trn,
                    epochs = 8,
                    validation_split = 0.1,
                    validation_steps = 10,
                    class_weight = {0 : 0.8 , 1 : 1.2})

# %% PLOTS
loss_acc_plots(history)

# %% EVALUATE
_, trn_acc = model.evaluate(d_trn, l_trn, verbose=0)
_, tst_acc = model.evaluate(d_tst, l_tst, verbose=0)
print(f"Train acc:{trn_acc} | Test acc:{tst_acc}")

# %% PERFORMANCE METRICS
tst_prob = model.predict(d_tst, verbose=0)
# tst_clas = model.predict_classes(d_tst, verbose=0) ## gives warning to use np.argmax
tst_clas = np.argmax(tst_prob, axis=1) 
tst_clas_oh = ohe.transform(tst_clas.reshape(-1,1)).toarray()


# %%
## get performance
pr, rc, fb, support = precision_recall_fscore_support(l_tst, tst_clas_oh)
print(f"Precision:{pr} | Recall:{rc} | FB:{fb} | Support:{support}")


# %% CHANGELOG


# %% TO DO
## Pr, Rc for pos class?? What's happenning?
## Try simple FNN?
## Update the normalization scheme - centered towards mean?
## Add more data (i.e. scRNA data, human (other models) data, etc.)


## NOTES
## why Loss is NaN - is zscore transformation culrit? Is activation required?

## HOW TO PROCESS LABELS - single, binarize or one-hot encode, and what value, activation and crossnetropy?
