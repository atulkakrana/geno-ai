#!/usr/bin/env python

# %% IMPORTS
import sys
import s3fs
import pandas as pd 
fs = s3fs.S3FileSystem(anon=False, profile_name="dips")

# %% IMPORTS
import rpy2.robjects as ro
import numpy as np
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

# %% READ FROM S3 - FPKM | IMPUTED
df_main_fpkm_imp = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.impute.tsv'),
                       sep = "\t" )
df_main_fpkm_imp.to_pickle("exprs_data_fpkm_impute.p", compression='infer', protocol=4)
df_main_fpkm_imp.head()

# %% READ FROM S3 - TPM
df_main_tpm = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.tsv'), sep = "\t") 
df_main_tpm.to_pickle("exprs_data_tpm.p", compression='infer', protocol=4)
df_main_tpm.head()

# %% READ FROM S3 - TPM | IMPUTED
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
    ## output
    xbool = False
    
    miss_cnts = df.isnull().sum(axis=1)
    df['empty_cells'] = miss_cnts
    mask   = df['empty_cells'] > 0
    df     = df[mask]
    nrows  = sum(mask)
    ncells = sum(list(df['empty_cells']))
    if nrows > 0:
        print(f"{nrows} rows with {ncells} empty cells\n")
        sys.exit()
        xbool = True
    else:
        print(f"No empty cells found in dataframe\n")
        pass

    return xbool

def transform_exprs(exprs_in, method = 'log'):
    '''
    get express matrix, should be in fload type;
    transforms matrix using different methods
    '''
    import numpy as np

    if method == 'log':
        no_zero_val = 0.000001
        exprs_tf = exprs_in.add(no_zero_val).apply(np.log10)
    else:
        print(f"{method} not implemented yet")
        sys.exit()

    return exprs_tf

def process_exprs_data(self, non_exprs_idxs, method = "log"):
    """
    the input is a dataframe with first two columns with
    gene identifiers, gene names, etc.

    the function was written to process data from here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6658352/
    """

    ## sanity check for missing data
    emp_bool = check_empty_cells(self)

    ## extract exprs data
    exprs      = self.drop(self.columns[non_exprs_idxs], axis=1)
    exprs_trfd = transform_exprs(exprs, method = method)

    ## regenrate dataframe;
    ## combine expression and other columns
    non_exprs_data = self.iloc[:,non_exprs_idxs]
    data_trfd = pd.concat([non_exprs_data,exprs_trfd], axis = 1)

    return data_trfd


#### PREPROCESS
# %% MAIN
non_exprs_idxs = [0,1] ## indexes for columns other than exprssion data i.e. gene info, etc.
data_trfd      = process_exprs_data(df_main_tpm_imp, non_exprs_idxs, method = 'log')




# %% DEV
# exprs = df_main_tpm_imp.iloc[:,2:]
exprs = df_main_tpm_imp.drop(df_main_tpm_imp.columns[[1, 2]], axis=1)


# %% TEST
df_main_fpkm.shape
df_main_fpkm.count()





# %%
#### CHANGELOG


# %%
#### TO DO
## 1. How to manage rows with no values, or value for some columns - check article?
