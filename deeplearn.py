#!/usr/bin/env python
## conda activate binf

## the deep learning
## note book for genoAI

# %% ENVIRONMENT
import os, datetime
# %load_ext tensorboard
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# %tensorboard --logdir logs/fit

# %% IMPORTS
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from traingen import prep_trainset
from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_recall_curve

import tensorflow as tf
from tensorflow.keras import layers, activations, models, metrics
from tensorflow.keras import losses, optimizers, callbacks, utils

# %% TOOLS
lb  = LabelBinarizer()
ohe = OneHotEncoder()

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

def pr_plot(cl_vec, prob_vec):
    '''
    inputs: 
    (a) one hot encoded class vectors form predictions
    (b) probabilitiy vectors from predictions 
    '''
    l = np.argmax(cl_vec, axis=1) ##  one hot (predicted) class vector
    p = np.max(prob_vec, axis=1)    ##  probabilities for each class
    precision, recall, thresholds = precision_recall_curve(l, p)
    plt = sns.lineplot(x=recall, y=precision, ci= 'sd')
    plt.set(xlabel="Recall", ylabel = "Precision")

    return None

def z_norm(x_arr):
    '''
    compute column-level z-score
    '''
    ## method 1
    # x     = x_arr - np.mean(x_arr, axis = 0)
    # y_arr = x/np.std(x_arr, axis = 0)

    ## method 2
    y_arr = stats.zscore(x_arr, axis=0, ddof=1)

    return y_arr

def log_norm(x_arr):
    '''
    log norm the expression arrays
    '''
    no_zero_val = 0.000001
    y_arr       = np.log10(x_arr + no_zero_val)

    return y_arr

def balance_train(data, labs, seed = None, balance = True):
    '''
    use imbalance-learn to balance the sets
    https://imbalanced-learn.org/stable/api.html#module-imblearn.under_sampling._prototype_selection
    
    for multiple features, get indexes, and use that to sample data
    https://stackoverflow.com/questions/60762538/how-to-get-sample-indices-from-randomundersampler-in-imblearn
    '''

    from collections import Counter 
    from random import randrange

    print(f'Input data:{Counter(labs)}')
    print(f"Balancing is {str(balance)}")

    if balance == True:
        ## set a seed
        if seed is None:
            seed = randrange(100)
        else:
            seed = int(seed)
        
        ## resample data
        # cnn  = CondensedNearestNeighbour(random_state=seed)
        rus  = RandomUnderSampler(random_state=seed)
        data_b, labs_b  = rus.fit_resample(data, labs)
        data_idxs       = rus.sample_indices_
    
    else:
        ## just pass as is
        data_b = np.copy(data)
        labs_b = np.copy(labs)
    
    print(f'Resampled:{Counter(labs_b)}')
    return data_b, labs_b, data_idxs

def prep_data(DATA_PKL, balance = True, seed = None):
    '''
    Reads the featurized data, balances, normalizes (if necessary),
    and ckecks for sanity
    '''
    ## Read Data Dict
    data_dct    = prep_trainset(DATA_PKL); data_dct.keys()
    data_exp    = data_dct['data_exp']
    data_pro    = data_dct['data_pro']
    labs_enc    = data_dct['labels']
    data_idxs   = np.expand_dims(np.arange(labs_enc.shape[0]), axis = 1)

    ## Get Balance Indexes
    data_idxs_a, labs_enc_b, data_idxs_b = balance_train(data_idxs, labs_enc, 
                                            seed = seed, 
                                            balance = balance)
    ## Get Balanced Data
    data_exp_b = data_exp[data_idxs_b]
    data_pro_b = data_pro[data_idxs_b]

    ## Normalize Data
    # data_exp_b     = log_norm(data_exp)
    data_exp_b   = z_norm(data_exp_b)

    ## Sanity Check
    if np.isnan(np.min(data_exp_b)):
        print("NaNs in data")
        sys.exit()

    if np.isnan(np.min(data_pro_b)):
        print("NaNs in data")
        sys.exit()

    return data_exp_b, data_pro_b, labs_enc_b

def shape_data(data_exp_b, data_pro_b, shape = "2D"):
    '''
    Reshape data to for downstream models
    '''

    if shape == '1D':
        ## Just exapand dims
        dexp = np.expand_dims(data_exp_b, axis = 2) ## For 1D CNN
        dpro = np.expand_dims(data_pro_b, axis = 2) ## For 1D CNN

    elif shape == '2D':
        ## Reshape to square form
        data_exp_norm_pad = np.pad(data_exp_b, ((0, 0), (0, 2)), mode='constant')
        data_pro_norm_pad = np.pad(data_pro_b, ((0, 0), (0, 4)), mode='constant')

        dexp = data_exp_norm_pad.reshape(data_exp_norm_pad.shape[:-1] + (12,8,1)) ## For 2D CNN
        dpro = data_pro_norm_pad.reshape(data_pro_norm_pad.shape[:-1] + (12,11,1)) ## For 2D CNN
    
    else:
        print(f"Value for 'Shape' argument not supported:'{shape}' - exiting")
        sys.exit()


    print(f"Original shape:{data_exp_b.shape}| New Shape:{dexp.shape}")
    print(f"Original shape:{data_pro_b.shape}| New Shape:{dpro.shape}")
    return dexp, dpro

def split_data(dexp, dpro, labs_norm):
    '''
    return simple startified split with class weights
    '''

    ## Generate Index Array
    ## (used because we have multiple features)
    splt_idxs   = np.arange(labs_norm.shape[0])
    trn_idx, tst_idx, l_trn, l_tst  = train_test_split(splt_idxs, labs_norm, 
                                                        test_size=0.1, stratify = labs_norm) 

    ## Fetch Train and Test Splits
    d_exp_trn = dexp[trn_idx]
    d_exp_tst = dexp[tst_idx]
    d_pro_trn = dpro[trn_idx]
    d_pro_tst = dpro[tst_idx]

    print(f"Train Shape:{d_exp_trn.shape} | {d_pro_trn.shape}")
    print(f"Test Shape :{d_exp_tst.shape} | {d_pro_tst.shape}")
    return d_exp_trn, d_pro_trn, d_exp_tst, d_pro_tst, l_trn, l_tst

def gen_class_wgts(l_trn):
    '''
    Generate class weights from labels;
    returns a dict
    '''

    class_wgts      = compute_class_weight('balanced', 
                        np.unique(np.argmax(l_trn, axis =1)), 
                        np.argmax(l_trn, axis =1))
    class_wgts_dct   = dict(enumerate(class_wgts, 0))

    print(f"Class weights:{class_wgts_dct}")
    return class_wgts_dct

def label_transform(labs_enc_b, method = 'ohe'):
    '''
    Input is encoded labels (generally balanced)
    Transform labels for the task
    '''

    if method == "binary":
        labs_norm = lb.fit_transform(labs_enc_b)
    
    elif method == "ohe":
        labs_norm = ohe.fit_transform(labs_enc_b.reshape(-1,1)).toarray()

    else:
        print(f"Method to transform labels is not supported:{labs_norm} - Exiting")
        sys.exit()

    return labs_norm


# %% PREP DATA ##############
#############################
# %% GET DATA [balanced]
data_exp_b, data_pro_b, labs_enc_b = prep_data(DATA_PKL, balance = True, seed = None)
labs_norm = label_transform(labs_enc_b, method = 'ohe')

# %% RSEHAPE DATA
dexp, dpro = shape_data(data_exp_b, data_pro_b, shape = "2D")

# %% SPLIT DATA
d_exp_trn, d_pro_trn, d_exp_tst, d_pro_tst, l_trn, l_tst = split_data(dexp, dpro, labs_norm)
class_wgts_dct = gen_class_wgts(l_trn)

# %% INSPECT DATA
# df_exp = pd.DataFrame(dexp)
# df_pro = pd.DataFrame(dpro)
# df_pro.describe()


#%% MODELS ##################
#############################
# %% MODELS
def simple_2d_cnn(data,labs):
    '''
    Simple (seqeunntial) NN for testing
    '''

    ## shapes
    di,dj,dk,dl = data.shape
    li,lj       = labs.shape
    print(f"Data Shape:  {data.shape}")
    print(f"Labels Shape:{labs.shape}")

    ## binary or multi-class
    if lj > 1:
        ## multi-class
        last_act = activations.softmax
        loss     = losses.categorical_crossentropy
    else:
        ## binary
        last_act = activations.sigmoid
        loss     = losses.binary_crossentropy

    ## layers and model
    l0 = layers.InputLayer((dj,dk,dl), name = "input")

    a1 = layers.Conv2D(16, (2,2), activation=activations.swish, name = "c2d_a1")
    a2 = layers.Conv2D(16, (2,2), activation=activations.swish, name = "c2d_a2")
    a3 = layers.MaxPool2D((2,2), name = "pool_a3")
    a4 = layers.Dropout(0.10, name = "drop_1")

    b1 = layers.Conv2D(32, (2,2), activation=activations.swish, name = "c2d_b1")
    b2 = layers.MaxPool2D((2,2), name = "pool_b2")
    b3 = layers.Dropout(0.10, name = "drop_2")

    l1 = layers.Flatten(name = "flat")
    l2 = layers.Dropout(0.10, name = "drop_3")

    l3 = layers.Dense(32, activation=activations.swish)
    l4 = layers.Dense(16, activation=activations.swish)
    lf = layers.Dense(lj,  activation=last_act)

    model = models.Sequential([l0, a1, a2, a3, a4, b1, b2, b3, l1, l2, l3, l4, lf])

    ## compile
    opt = optimizers.Adam(learning_rate=0.0000000001)
    model.compile(optimizer = opt,
              loss          = loss,
              metrics       = ['accuracy']) ## metrics.RecallAtPrecision(precision=0.5);PrecisionAtRecall(recall=0.5)

    return model

def multi_inp_2D_CNN():



    return None

def simple_1D_cnn(data, labs):
    '''
    Very simple 1D CNN

    Shape: https://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape/43236878
    '''
    d1,d2,d3 = data.shape
    l1,l2 = labs.shape
    print(f"Data Shape:{data.shape}")
    print(f"Labels Shape:{labs.shape}")

    ## decisions
    if l2 > 1:
        last_act = activations.softmax
        loss     = losses.categorical_crossentropy
    else:
        last_act = activations.sigmoid
        loss     = losses.binary_crossentropy


    # input = layers.InputLayer(input_shape = (95,1), name ="input") ## this didn't work
    input = tf.keras.Input(shape=(d2,d3))

    x = layers.Conv1D(16, 3, activation=activations.relu, name = "c2d_a1")(input)
    x = layers.MaxPool1D(2, name  = "pool_a2")(x)
    x = layers.Dropout(0.25, name = "drop_1")(x)

    x = layers.Conv1D(16, 3, activation=activations.relu, name = "c2d_b1")(x)
    x = layers.MaxPool1D(2, name  = "pool_b2")(x)
    x = layers.Dropout(0.25, name = "drop_2")(x)

    d = layers.Flatten(name       = "flat")(x)

    d = layers.Dense(32, activation=activations.swish)(d)
    d = layers.Dense(16, activation=activations.swish)(d)
    output = layers.Dense(l2,  activation=last_act)(d)

    model = tf.keras.Model(inputs=input, outputs=output)

    opt = optimizers.Adam(learning_rate=0.0000000001)
    model.compile(optimizer = opt,
              loss          = loss,
              metrics       = ['accuracy']) ## metrics.RecallAtPrecision(precision=0.5);PrecisionAtRecall(recall=0.5)

    return model

# %% SUMMARY
# model = simple_1D_cnn(d_exp_trn, l_trn)
model = simple_2d_cnn(d_exp_trn, l_trn)
model.summary()

dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi= 64)

# %% FIT
history = model.fit(x = d_exp_trn,
                    y = l_trn,
                    epochs = 8,
                    batch_size = 32,
                    validation_split = 0.1,
                    validation_steps = 5,
                    class_weight     = class_wgts_dct) ## callbacks=[tensorboard_callback]




# %% PERFORMANCE ############
#############################
# %% BASIC PLOT
loss_acc_plots(history)

# %% BASIC EVAL
_, trn_acc = model.evaluate(d_exp_trn, l_trn, verbose=0)
_, tst_acc = model.evaluate(d_exp_tst, l_tst, verbose=0)
print(f"Train acc:{trn_acc} | Test acc:{tst_acc}")

# %% EVALUATE
def evaluate(d_exp_tst, l_tst):
    '''
    Evaluation of model on test data
    '''
    ## Get Prediction Probs
    ## https://datascience.stackexchange.com/questions/27153/how-to-get-predicted-class-labels-in-convolution-neural-network/27756
    tst_probs = model.predict(d_exp_tst, verbose=0)
    # tst_clas = model.predict_classes(d_tst, verbose=0) ## gives warning to use np.argmax

    ## Get Classes
    if l_tst.shape[1] == 1:
        tst_class = (tst_probs < 0.5).astype(np.int)
    else:
        tst_class = np.argmax(tst_probs, axis=1) 
        tst_class = ohe.transform(tst_class.reshape(-1,1)).toarray()

    ## Get Performance For All Labels
    print(classification_report(l_tst, tst_class, labels=[0,1]))

    return tst_class, tst_probs

# %% PERFORMANCE
tst_class, tst_probs = evaluate(d_exp_tst, l_tst)
pr_plot(tst_class, tst_probs)



# %% CHANGELOG
## v01

## v02
## revert to binary classifer from one hot encode


# %% TO DO
## Pr, Rc for pos class?? What's happenning?
## Try simple FNN?
## Update the normalization scheme - centered towards mean?
## Add more data (i.e. scRNA data, human (other models) data, etc.)


## NOTES
## 1. why Loss is NaN - is zscore transformation culrit? Is activation required?
## Normalize original (CPM) data using zscore approach and not log data - take origonal data to pickle

## 2. HOW TO PROCESS LABELS - single, binarize or one-hot encode, and what value, activation and crossnetropy?
## 3. Primer on Optimizers and activations


