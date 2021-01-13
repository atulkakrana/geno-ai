#!/usr/bin/env python

## the deep learning
## note book for genoAI

# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from traingen import prep_trainset
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

# %% GET DATA
data_dct = prep_trainset(DATA_PKL)
data_dct.keys()

# %% PROCESS
data_exp     = data_dct['data_exp']
labs_enc     = data_dct['labels']

# %% PAD AND RESHAPE
data_exp_pad = np.pad(data_exp, ((0, 0), (0, 1)), mode='constant')
print(f"Original shape:{data_exp.shape} | New Shape:{data_exp_pad.shape}")

dexp = data_exp_pad.reshape(data_exp_pad.shape[:-1] + (12,8,1))
dexp.shape

# %% SPLITS
d_trn, d_tst, l_trn, l_tst = train_test_split(dexp, labs_enc, test_size=0.1) 

# %% MODELS
def simple_cnn():
    '''
    Simple (seqeunntial) NN for testing
    '''

    l0 = layers.InputLayer((12,8,1), name = "input")

    a1 = layers.Conv2D(24, (2,2), activation=activations.relu, name = "c2d_a1")
    a2 = layers.MaxPool2D((2,2), name = "pool_a2")

    b1 = layers.Conv2D(12, (2,2), activation=activations.relu, name = "c2d_b1")
    b2 = layers.MaxPool2D((2,2), name = "pool_b2")

    l1 = layers.Flatten(name = "flat")
    l2 = layers.Dropout(0.5, name = "drop")

    l3 = layers.Dense(24, activation=activations.swish)
    l4 = layers.Dense(12, activation=activations.swish)
    lf = layers.Dense(1,  activation=activations.sigmoid)

    model = models.Sequential([l0, a1, a2, b1, b2, l1, l2, l3, l4, lf])

    return model


# %% SUMMARY
## summary
model = simple_cnn()
model.summary()

## plot
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


# %% COMPILE
model.compile(optimizer ='rmsprop',
              loss      = losses.binary_crossentropy,
              metrics   = ['accuracy'])

# %% Fit
history = model.fit(x = d_trn,
                    y = l_trn,
                    epochs = 8,
                    validation_split = 0.1,
                    validation_steps = 10,
                    class_weight = {0 : 0.7 , 1 : 1.3})




# %% PLOTS
loss_acc_plots(history)

# %% EVALUATE
_, trn_acc = model.evaluate(d_trn, l_trn, verbose=0)
_, tst_acc = model.evaluate(d_tst, l_tst, verbose=0)
print(f"Train acc:{trn_acc} | Test acc:{tst_acc}")

# %% PERFORMANCE METRICS
tst_prob = model.predict(d_tst, verbose=0)
tst_clas = model.predict_classes(d_tst, verbose=0)

## get performance
pr, rc, fb, support = precision_recall_fscore_support(l_tst, tst_clas)
print(f"Precision:{pr} | Recall:{rc} | FB:{fb} | Support:{support}")


# %% CHANGELOG


# %% TO DO
## Pr, Rc for pos class?? What's happenning?
## Try simple FNN?
## Update the normalization scheme - centered towards mean?
## Add more data (i.e. scRNA data, human (other models) data, etc.)
