#!/usr/bin/env python

## Generate Training Data
## Config in YAML files

# %% IMPORTS
from dlcore import *


# %% INPUTS
DATA_PKL = "train_data.p"

# %% FUNCTIONS
def prep_trainset(DATA_PKL):
    '''
    Generates training/test set for ML/DL;
    takes output of DLCORE
    '''

    data_dct     = pickle.load( open( DATA_PKL, "rb" ) )
    labels_enc   = encode_labels(data_dct)
    data_exprs   = data_dct['exp_data']



    return None

# %% MAIN - INTERACTIVE
tset = prep_trainset()

# %% TEST


# %% MAIN


# %% RUN
if __name__ == "__main__":
    main()
    pass

# %% CHANGELOG
## v01 [12/26/2020]