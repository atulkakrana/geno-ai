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
df_all_labs = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/all_gene_disease_associations.tsv', sep="\t")
df_all_labs.head()

df_cur_labs = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/curated_gene_disease_associations.tsv', sep="\t")
df_cur_labs.head()

df_dis_attr  = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/disease_mappings_to_attributes.tsv', sep="\t")
df_dis_attr.head()


# %% CUI To Diseases Class Mappings (i.e. labels)
## https://en.wikipedia.org/wiki/List_of_MeSH_codes
## From attributes file choose diseasesClass == C16 (for congenital diseases)
## then for each CUI identify the broader diseases class (i.e. labels)
##              Use diseasesClassNameMSH (from diseases attr files)
## then map these labels to labs file i.e. assign broader diseases class to each CUI which will serve as label




# %% CHANGELOG




# %% TO DO

## Which set of gene list to use - curated?
## Which diseases labels to choose and how?