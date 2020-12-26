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

    labs_dct, unmap_lst = update_labs_to_ensembl(LAB_PKL, species)
    lab_data_pkl        = gen_data_labels(DATA_FL, labs_dct)



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