#!/usr/bin/env python

## Prepare labesl for genes
## NOTE: here we have tried to use 
#### pandas as much we can so the
#### code is not that efficient

# %% ENV VARIABLES
from dlcore import DATA_FL
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
config  = yaml.load(open('prepare_labels.yaml', 'r'), Loader=yaml.FullLoader)
DATA_FL = config['user']['data_file']
LABS_FL = config['user']['labs_file']
ATTR_FL = config['user']['attr_file']
CLASS   = config['dev']['class_type']

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

    out_df = pd.read_csv(s3_fl_path, sep=sep)
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

def gen_dis_labs_dct(cur_labs_df):
    '''
    For every gene provide all listed 
    MSH diseases codes as labels

    Used Pandas here just for the sake 
    of practice
    '''

    ## outputs
    labs_dct = {}

    ## inputs
    grdf = cur_labs_df.groupby("geneSymbol")
    gdct = grdf.groups
    gkey = gdct.keys()
    gval = gdct.values()

    ## iterate over each group key
    ## and collect MSH codes
    for k,v in zip(gkey,gval):
        # print(f"\ngene:{k} | indexes{list(v)}")
        vals = []
        
        for idx in list(v):
            astr = cur_labs_df.loc[idx]['diseaseClass']
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
    elem_cnts = [len(x) for x in labs_dct.values()]
    ax1        = sns.countplot(x=elem_cnts); plt.show()
    
    elem_type_cnts = list(itertools.chain.from_iterable(labs_dct.values()))
    ax2        = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index)
    plt.show()

    print(f"Elements in labels dct:{len(labs_dct)}")
    return labs_dct

def process_labs(indct):
    '''
    remove certain diseases classes based on their
    low occurance in disgeneset, and based n the 
    manual curation of MSH diseases class codes
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
    for k,v in labs_dct.items():
        if binary_tar.intersection(v):
            labs_bin_dct[k] = "pos"
        else:
            labs_bin_dct[k] = "neg"

    ## visualize
    elem_type_cnts = list(itertools.chain.from_iterable(labs_dct.values()))
    ax1            = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index); plt.show()

    elem_type_cnts = list(labs_bin_dct.values())
    ax2            = sns.countplot(y=elem_type_cnts, order = pd.Series(elem_type_cnts).value_counts().index); plt.show()

    ## write pickles
    pickle.dump( labs_dct, open( "labs_dct.p", "wb" ) )
    pickle.dump( labs_bin_dct, open( "labs_bin_dct.p", "wb" ) )

    print(f"Elements in labels dct:{len(labs_dct)}")
    return labs_dct, labs_bin_dct

def prepare_labs(DATA_FL, labs_dct):
    '''
    reads keys from final processed data (TSV) file, and 
    generates a list of labels for ML, plus
    writes a labelled dataset for manual inspection
    '''

    data = pd.read_csv(DATA_FL, sep = "\t")
    ids  = data.iloc[:, 0].to_list()



    

    return None

# %% MAIN - INTERACTIVE
## %%capture
# labs_df, cur_labs_df, dis_attr_df = read_lab_data()
# filt_attr_df  = dis_attr_mask_gen(dis_attr_df)
# gene_labs_dct = gen_dis_labs_dct(cur_labs_df)
# fin_labs_dct, bin_labs_dct = process_labs(gene_labs_dct)

# %% TEST


# %% DEV
data = pd.read_csv(DATA_FL, sep = "\t")
data.set_index(data.columns[0], inplace = True)
# ids  = data.iloc[:, 0].to_list()


# %% MAIN
def main():
    cur_labs_df    = read_s3_df(LABS_FL, sep = "\t")
    gene_labs_dct  = gen_dis_labs_dct(cur_labs_df)
    fin_labs_dct, bin_labs_dct = process_labs(gene_labs_dct)
    if CLASS   = "binary":
        labs_dct = prepare_labs(bin_labs_dct)
    elif CLASS = "multi":
        prepare_labs(bin_labs_dct)
        labs_dct = prepare_labs(fin_labs_dct)
    else:
        print(f"Classificstion method '{CLASS}' is not yet implemented")
        sys.exit()


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

## Which set of gene list to use - curated?
## Which diseases labels to choose and how?