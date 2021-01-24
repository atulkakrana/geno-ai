#!/usr/bin/env python

## Reads processed data
## and labels and prepares
## in format required for 
## deep-learning

# %% IMPORTS
from dlcore import *

# %% INPUTS
DATA_PKL = "train_data.p"

# %% FUNCTIONS
def prep_trainset(DATA_PKL):
    '''
    DATA_PKL: holds different data, and labels
    as equal length arrays and lists. Generated
    by 'dlcore.py'

    Generates training/test set for ML/DL;
    takes output of DLCORE.
    '''

    data_dct    = pickle.load( open( DATA_PKL, "rb" ) )
    labs_enc    = encode_labels(data_dct)
    data_exp    = data_dct['exp_data']
    data_pro    = data_dct['pro_data']

    ## off for deep learning
    payload     = { 'data_exp':data_exp, 
                    'data_pro':data_pro,
                    'labels':labs_enc
                    }
    return payload

# %% MAIN - INTERACTIVE
# data_dct = prep_trainset(DATA_PKL)

# %% TEST

# %% MAIN
def main():
    payload = prep_trainset(DATA_PKL)
    return None

# %% RUN
if __name__ == "__main__":
    main()
    pass

# %% CHANGELOG
## v01 [12/26/2020]