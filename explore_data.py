#!/usr/bin/env python


#### IMPORTS
# %% 
import s3fs
import botocore
import pandas as pd 
# s3 = s3fs.S3FileSystem(anon=False, profile_name="dips")
session = botocore.session.Session(profile='dips')
fs = s3fs.core.S3FileSystem(anon=False, session=session)


# %%
df_main  = pd.read_csv(
            fs.open('s3://lachke-lab-data/work/0.geno-ai/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv') )

# %%
with fs.open('s3://lachke-lab-data/work/0.geno-ai/E-MTAB-6798/E-MTAB-6798.xlsx') as f:
    df = pd.read_excel(f)

# %%
