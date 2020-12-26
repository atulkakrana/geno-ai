#!/usr/bin/env python

# %% IMPORTS
import s3fs
import pandas as pd 
fs = s3fs.S3FileSystem(anon=False, profile_name="dips")

# %% IMPORTS
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# %% HELPERS
def df_py_to_r(df_py):
    '''
    Converts python to R data frame
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df_py)
    return df_r


#### INPUTS
###########
# %%  READ FROM S3 - FPKM
# df_main  = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
df_main_fpkm = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv'),
                       sep = "\t" )
df_main_fpkm.head()

# %% READ FROM S3 - CPM
df_main_cpm = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.tsv'), sep = "\t" )
df_main_cpm.head()

# %% READ LOCAL
# df_main_fpkm  = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
# df_main_cpm   = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.tpms.tsv', sep="\t")
# df_main_fpkm.head()


#### PREPROCESS


# %%
#### CHANGELOG


# %%
#### TO DO
## 1. How to manage rows with no values, or value for some columns - check article?
