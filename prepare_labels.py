#!/usr/bin/env python

## Prepare labesl for genes
## NOTE: here we have tried to use 
#### pandas as much we can so the
#### code is not that efficient

# %% ENV VARIABLES
from dlcore import *
import os
import sys
from rpy2.robjects.packages import data
import s3fs
import seaborn as sns
HOME = os.getenv("HOME")
FS   = s3fs.S3FileSystem(anon=False, profile_name="dips")
sns.set_theme(style="darkgrid")


# %% IMPORTS
import itertools
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt

## USER SETTINGS
config      = yaml.load(open('prepare_labels.yaml', 'r'), Loader=yaml.FullLoader)
DATA_FL     = config['user']['data_file']
LABS_FL     = config['user']['labs_file']
ATTR_FL     = config['user']['attr_file']
LABS_ALL_FL = config['user']['labs_all_file']
CLASS       = config['dev']['class_type']

# %% HELPERS
def config_reader(YAML):
    '''
    takes config from YAML files
    '''
    ## imports
    import yaml

    with open(YAML, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def frq_plot(cnts_lst):
    '''
    Generates Frequancy plot from a 
    list of counts
    '''

    ax = sns.countplot(cnts_lst)
    return None

def read_s3_df(s3_fl_path, sep = "\t"):
    '''
    read the disgenet gene-diseases labels and the 
    attributes file
    '''
    # labs_df = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/all_gene_disease_associations.tsv'), sep="\t")
    # labs_df.head()

    # cur_labs_df = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/curated_gene_disease_associations.tsv'), sep="\t")
    # cur_labs_df.head()

    # dis_attr_df = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/disgenet/disease_mappings_to_attributes.tsv'), sep="\t")
    # dis_attr_df.head()

    out_df = pd.read_csv(FS.open(s3_fl_path), sep=sep)
    return out_df

# %% DISEASES GENE LABELS
def dis_attr_mask_gen(attr_df):
    '''
    Extracts diseases rows that correspond to the 
    MSH codes provides pos_codes
    '''
    ## Filters
    pos_codes = ["C16"]

    ## Outputs
    mask = []

    ## Inputs
    class_df = attr_df[['diseaseClassMSH']]
    class_df['index'] = class_df.index
    anarr = class_df.values
    print(f"Classes array:{anarr[:10]}\n")

    ## Generate mask for rows
    ## that correspond to a congenital 
    ## diseases
    for i in anarr:
        # print(f"ent:{i}")
        val, idx = i
        if pd.isna(val):
            mask.append(False)
        else:
            codes     = val.split(";")
            tmp_bools = []
            for code in codes:
                if code in pos_codes:
                    tmp_bools.append(True)
                else:
                    tmp_bools.append(False)
            
            if sum(tmp_bools) > 0:
                mask.append(True)
            else:
                mask.append(False)

    ## filter curated labels for 
    ## congenital diseases
    filt_attr_df = attr_df[mask]

    print(f"Total rows:{len(anarr)} | Len of mask:{len(mask)}")
    print(f"To retain:{sum(mask)}   | Retained:{len(filt_attr_df.index)}")
    return filt_attr_df

def gen_dis_labs_dct(labs_df):
    '''
    For every gene provide all listed 
    MSH diseases codes as labels

    Used Pandas here just for the sake 
    of practice
    '''

    ## outputs
    labs_dct = {}

    ## inputs
    grdf = labs_df.groupby("geneSymbol")
    gdct = grdf.groups
    gkey = gdct.keys()
    gval = gdct.values()

    ## iterate over each group key
    ## and collect MSH codes
    for k,v in zip(gkey,gval):
        # print(f"\ngene:{k} | indexes{list(v)}")
        vals = []
        
        for idx in list(v):
            astr = labs_df.loc[idx]['diseaseClass']
            if pd.isna(astr):
                pass
            else:
                alst = astr.split(";")
                vals.extend(alst)
                # print(f"codes:{alst}")
        
        vals = set(vals)
        # print(f"Adding {len(vals)} diseases codes")
        labs_dct[k] = vals

    ## visualize
    elem_cnts   = [len(x) for x in labs_dct.values()]
    ax1         = sns.countplot(x=elem_cnts); plt.show()
    
    elem_type_cnts  = list(itertools.chain.from_iterable(labs_dct.values()))
    ax2             = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index)
    plt.show()

    print(f"Elements in labels dct:{len(labs_dct)}")
    return labs_dct

def process_labs(indct, gene_all_labs_dct, mode = 'binary'):
    '''
    remove certain diseases classes based on their
    low occurance in disgeneset, and based n the 
    manual curation of MSH diseases class codes
    
    binary: two labels and None;
    mclass, mlabel: all labels as-is and unlabelled as None or 'None'
    '''
    ## outputs
    labs_dct     = {}
    labs_bin_dct = {}

    ## inputs
    remove_set = set(['C01', 'C07', 'C09', 'C16', 'C21', 'C22', 'C23', 'C24' , 'C26']) ## these will be removed
    binary_tar = set(['C11',]) ## these will form positive category

    ## generate labels for
    ## multi-label classification
    for k,v in indct.items():
        nset = v-remove_set
        labs_dct[k] = nset
    
    ## generate labels for binary classification
    ## binarize labels, any gene
    ## with eye diseases gets pos
    ## labels and others get neg
    ## genes with eye disease along
    ## with others also get pos label
    likely_pos  = []
    pos         = []
    neg         = []
    if mode == 'binary':
        for k,v in labs_dct.items():
            if binary_tar.intersection(v):
                label = "pos"
                pos.append(k)
            else:
                label = "neg"

                ## cleanup negative using the
                ## labels from text mining; i.e.
                ## silver set labels. the negavtive
                ## set should not have target labels
                tmp_dis_set = gene_all_labs_dct.get(k)
                # print(f"Diseases codes from full labels set:{tmp_dis_set}")
                if binary_tar.intersection(tmp_dis_set):
                    ## evidences from text mining
                    ## show gene assosiation with
                    ## positive class, hence it's
                    ## likely positive, remove from
                    ## negative list and mark unlabelled
                    print(f"Likely Positive gene:{k}")
                    likely_pos.append(k)
                    label = None
                else:
                    neg.append(k)
                    pass
            
            ## assign final label
            if label:
                labs_bin_dct[k] = label
    else:
        print(f"The mode:{mode} is not understood")
        sys.exit(1)
        pass

    print(f"Postive:{len(pos)} | Likely Positive (removed from negative):{len(likely_pos)} | Negative:{len(neg)}")

    ## visualize
    elem_type_cnts = list(itertools.chain.from_iterable(labs_dct.values()))
    ax1            = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index); plt.show()

    elem_type_cnts = list(labs_bin_dct.values())
    ax2            = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index); plt.show()

    ## write pickles
    pickle.dump( labs_dct, open( "labs_dct.p", "wb" ) )
    pickle.dump( labs_bin_dct, open( "labs_bin_dct.p", "wb" ) )

    print(f"Elements in labels dct:{len(labs_dct)} | binary_dct:{len(labs_bin_dct)}")
    return labs_dct, labs_bin_dct, pos, neg, likely_pos

# %% MAIN - INTERACTIVE
# %%capture
## curated labels
cur_labs_df    = read_s3_df(LABS_FL, sep = "\t")
gene_labs_dct  = gen_dis_labs_dct(cur_labs_df)

## all lables including text mining
all_labs_df         = read_s3_df(LABS_ALL_FL, sep = "\t")
gene_all_labs_dct   = gen_dis_labs_dct(all_labs_df)

## generate final labels
fin_labs_dct, bin_labs_dct, pos, neg, likely_pos = process_labs(gene_labs_dct, gene_all_labs_dct, mode = 'binary')

# %% TEST
likely_pos[1:10]

# %% DEV
# data = pd.read_csv(DATA_FL, sep = "\t")
# data.set_index(data.columns[0], inplace = True)
# ids  = data.iloc[:, 0].to_list()


# %% MAIN
def main():
    ## curated labels
    cur_labs_df         = read_s3_df(LABS_FL, sep = "\t")
    gene_labs_dct       = gen_dis_labs_dct(cur_labs_df)

    ## all lables including text mining
    all_labs_df         = read_s3_df(LABS_ALL_FL, sep = "\t")
    gene_all_labs_dct   = gen_dis_labs_dct(all_labs_df)

    ## generate final labels
    fin_labs_dct, bin_labs_dct, pos, neg, likely_pos = process_labs(gene_labs_dct, gene_all_labs_dct, mode = 'binary')

    return None

# %% 
if __name__ == "__main__":
    main()
    pass


# %% CHANGELOG
## V01 [12-26-2020]
## added functions to read data from S3
## added functions to extract MSH diseases codes as labels
## added function to clean labels, and generate gene-diseases labels dicts


# %% TO DO
## CUI To Diseases Class Mappings (i.e. labels)
## https://en.wikipedia.org/wiki/List_of_MeSH_codes
## From attributes file choose diseasesClass == C16 (for congenital diseases)
## then for each CUI identify the broader diseases class (i.e. labels)
##              Use diseasesClassNameMSH (from diseases attr files)
## then map these labels to labs file i.e. assign broader diseases class to each CUI which will serve as label

## Expand positive gene list - 
##      include genes with target category (C11) from all labelsfile
##      specifically those with 'HPO' label under the 'source' column
##      also check what other sources can be included 