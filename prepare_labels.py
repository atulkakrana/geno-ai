#!/usr/bin/env python

## prepare labesl for genes
## how to select high-confidence labelled gene set?
## what will be the split proportion?

# %% ENV VARIABLES
import os, sys
import s3fs
HOME = os.getenv("HOME")


# %% IMPORTS
import os
import pandas as pd



# %% DATA
df_main  = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/all_gene_disease_associations.tsv', sep="\t")
df_main.head()

# %%