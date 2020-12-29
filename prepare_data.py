#!/usr/bin/env python

# %% IMPORTS
import sys
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
df_main_fpkm.to_pickle("exprs_data_fpkm.p", compression='infer', protocol=4)
df_main_fpkm.head()

# %% READ FROM S3 - FPKM | IMPUTE
df_main_fpkm_imp = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.impute.tsv'),
                       sep = "\t" )
df_main_fpkm_imp.to_pickle("exprs_data_fpkm_impute.p", compression='infer', protocol=4)
df_main_fpkm_imp.head()

# %% READ FROM S3 - TPM
df_main_tpm = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.tsv'), sep = "\t") 
df_main_tpm.to_pickle("exprs_data_tpm.p", compression='infer', protocol=4)
df_main_tpm.head()

df_main_tpm_imp = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.impute.tsv'), sep = "\t") 
df_main_tpm_imp.to_pickle("exprs_data_tpm_impute.p", compression='infer', protocol=4)
df_main_tpm_imp.head()

# %% READ LOCAL
# df_main_fpkm  = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
# df_main_cpm   = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.tpms.tsv', sep="\t")
# df_main_fpkm.head()


# %% FUNCTIONS
def check_empty_cells(df):
    '''
    count empty cells in exprs data,
    and add counts as last column
    '''
    miss_cnts = df.isnull().sum(axis=1)
    df['empty_cells'] = miss_cnts
    mask   = df['empty_cells'] > 0
    df     = df[mask]
    nrows  = sum(mask)
    ncells = sum(list(df['empty_cells']))
    if nrows > 0:
        print(f"{nrows} rows with {ncells} empty cells\n")
        sys.exit()
    else:
        print(f"No empty cells found in dataframe\n")
        pass

    return None

#### PREPROCESS
# %% MAIN
check_empty_cells(df_main_tpm_imp)
check_empty_cells(df_main_tpm)


# %% DEV


# %% TEST
df_main_fpkm.shape
df_main_fpkm.count()





# %%
#### CHANGELOG


# %%
#### TO DO
## 1. How to manage rows with no values, or value for some columns - check article?
