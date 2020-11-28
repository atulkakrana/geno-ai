#!/usr/bin/env python

#### IMPORTS
# %% 
import s3fs
import botocore
import pandas as pd 
fs = s3fs.S3FileSystem(anon=False, profile_name="dips")


# %%  READ FROM S3
# df_main  = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
df_main  = pd.read_csv(fs.open('s3://lachke-lab-data/work/0.geno-ai/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv'),
                       sep = "\t" )
df_main.head()

# %% READ LOCAL
df_main  = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
df_main.head()
