#!/usr/bin/env python

## Generate Training Data
## Config in YAML files

# %% IMPORTS
from dlcore import *


# %% INPUTS
LAB_PKL = "labs_bin_dct.p" ## could be binary labels, multi-label or multi-class from "prepare_labels.py"
DATA_FL = "exprs_data_cpm.p"

# %% FUNCTIONS
def prep_trainset(LAB_PKL):
    '''
    Generates training/test set for ML/DL
    '''

    data_dct     = pickle.load( open( "train_data.p", "rb" ) )
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