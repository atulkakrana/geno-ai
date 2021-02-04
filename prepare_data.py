#!/usr/bin/env python

# %% IMPORTS
import sys
import s3fs
import pandas as pd 
FS = s3fs.S3FileSystem(anon=False, profile_name="dips")

# %% IMPORTS
import rpy2.robjects as ro
import numpy as np
import yaml
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri
from seq_embeddings import fasta_reader, process_seqs
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from fetchPromoters import add_flanks, filter_flanked, extract_seqs, clean_bed, gene_level_bed

#%% USER SETTINGS
config          = yaml.load(open('prepare_data.yaml', 'r'), Loader=yaml.FullLoader)
DATA_FL         = config['user']['data_file']
GENE_BED_FL     = config['user']['genes_bed']
CHR_SIZE_FL     = config['user']['chr_sizes']
GENOME_ASSM     = config['user']['genome_fl']

# %% HELPERS
def df_py_to_r(df_py):
    '''
    Converts python to R data frame
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df_py)
    return df_r

def data_reader(fl_path, sep = "\t", local = False):
    '''
    reads imputed expression data
    '''

    # ## READ FROM S3 - FPKM
    # # df_main  = pd.read_csv('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
    # df_main_fpkm = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.tsv'),
    #                     sep = "\t" )
    # df_main_fpkm.to_pickle("exprs_data_fpkm.p", compression='infer', protocol=4)
    # df_main_fpkm.head()

    # # %% READ FROM S3 - FPKM | IMPUTED
    # df_main_fpkm_imp = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.fpkms.impute.tsv'),
    #                     sep = "\t" )
    # df_main_fpkm_imp.to_pickle("exprs_data_fpkm_impute.p", compression='infer', protocol=4)
    # df_main_fpkm_imp.head()

    # # %% READ FROM S3 - TPM
    # df_main_tpm = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.tsv'), sep = "\t") 
    # df_main_tpm.to_pickle("exprs_data_tpm.p", compression='infer', protocol=4)
    # df_main_tpm.head()

    # # %% READ FROM S3 - TPM | IMPUTED
    # df_main_tpm_imp = pd.read_csv(FS.open('s3://lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/E-MTAB-6798-query-results.tpms.impute.tsv'), sep = "\t") 
    # df_main_tpm_imp.to_pickle("exprs_data_tpm_impute.p", compression='infer', protocol=4)
    # df_main_tpm_imp.head()

    # ## READ LOCAL
    # # df_main_fpkm  = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.tsv', sep="\t")
    # # df_main_cpm   = pd.read_csv('/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.tpms.tsv', sep="\t")
    # # df_main_fpkm.head()

    if local:
        data_df = pd.read_csv(fl_path, sep=sep)
    else:
        data_df = pd.read_csv(FS.open(fl_path), sep=sep)
    
    return data_df

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

def process_exprs_data(self, non_exprs_idxs, id_col = 0, method = "log"):
    """
    the input is a dataframe with first two columns with
    gene identifiers, gene names, etc.

    the function was written to process data from here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6658352/
    """

    ## imports
    import pickle as pkl
    import pandas as pd 

    ## output
    outtsv  = "data_imp_trfd.tsv"
    outpkl1 = "data_imp_dct.p"
    outpkl2 = "%s_dct.p" % outtsv.rpartition(".")[0]

    ## sanity check for missing data
    emp_bool   = check_empty_cells(self)

    ## extract exprs data
    # print(f"exprs:{self.shape}")
    exprs      = self.drop(['empty_cells'], axis = 1)
    exprs      = exprs.drop(exprs.columns[non_exprs_idxs], axis=1)
    exprs_trfd = transform_exprs(exprs, method = method)
    # exprs_array= exprs_trfd.to_numpy()
    print(f"exprs:{exprs.shape} | trfd:{exprs_trfd.shape}")
    # print(f"colnames:{exprs.columns}")

    ## plot exprs data
    fig = exprs_trfd.plot.box(figsize=(20,8), rot = 90).get_figure()
    fig.savefig('log_norm.pdf')

    ## regenrate dataframe;
    ## combine expression and other columns
    non_exprs_data  = self.iloc[:,non_exprs_idxs]
    data_trfd       = pd.concat([non_exprs_data,exprs_trfd], axis = 1)

    ## generate dictionary of expression
    ids             = list(data_trfd.iloc[:, id_col])
    data_arr        = exprs_trfd.values
    
    ## sanity check
    if len(ids)     == data_arr.shape[0]:
        data_trfd_dct = {k:v for k,v in zip(ids,data_arr)}
        data_dct      = {k:v for k,v in zip(ids,exprs.values)}
        print(f"Length of IDs:{len(ids)} | Unique IDs:{len(set(ids))} | Len of Dict:{len(data_trfd_dct)}")
    else:
        print("Length of IDs do not match size of expression array")
        sys.exit()

    ## write processed dataframe
    data_trfd.to_csv(outtsv, sep="\t", index=False)
    pkl.dump(data_dct,      open(outpkl1, 'wb'))
    pkl.dump(data_trfd_dct, open(outpkl2, 'wb'))

    return data_trfd

def fetch_promoters(genes_bed, chr_sizes, genome_fl, flank = 500):
    '''
    fetch promoter fasta file from given 
    genes bed files and output a dct of 
    cleaned upnames as keys and seqs as values

    for featured bed files see fetchPromoters.R (promoters from biomart);
    please note that bed files from UCSC table browser diidn't include
    a large number of genes in our expression data (from Ensembl 71, 2013)
    '''
    
    flanked_bed     = add_flanks(genes_bed, chr_sizes, flank)       ## extend coordinates to include flanks
    _, filtered_bed = filter_flanked(flanked_bed)                   ## select correct flanked coords based on + and - strand
    outfasta        = extract_seqs(filtered_bed, genome_fl)         ## extracts flanked coords seq in correct orientation
    _,outfeats      = process_seqs(outfasta, method = "ok", out_format = "fasta") ## Segment the seqeunces to generate embeddings 
    
    print(f"\nPromoters Fetched")
    print(f"Flanked seqeunces written to:{outfasta}")
    print(f"Segmented seqeunces written to:{outfeats}")
    return outfeats

def read_seq_features(fas_fl_lst, feat_lst):
    '''
    this function reads seqeunces for features;
    each feature is provided in seprate file such 
    5'UTR, 3'UTR, etc. and a corresponding feature
    names (i.e. gene, promotoe, UTR, etc)

    OUTPUT: dict of dicts i.e. (dict of promoter seqeunces, dict of gene seqeunces, etc.)
    '''
    print(f"\n#### Fn: Read and Process FASTA feature Files ########")

    ## output
    fasdct = {}
    outpkl = "dna_feats.pkl"

    ## sanity check
    if len(fas_fl_lst) == len(feat_lst):
        pass
    else:
        print(f"fasta files:{len(fas_fl_lst)} | feats labels:{len(feat_lst)}")
        print(f"the input list lengths should be same - exiting")
        sys.exit()

    for fl, feat in zip (fas_fl_lst, feat_lst):
        ## read and cleans
        ## fasta keys i.e.
        ## normalize to map with 
        ## expression data
        tmp_dct = {}
        fasta   = fasta_reader(fl)

        for head,seq in fasta.items():
            k = head.split(".",1)[0].replace("ENSMUST","ENSMUSG") ## incase seq extraction used UCSC table browswer which returns transcript ids
            k = k.split(":",1)[0] ## extract name from bedtools generated fasta header (ENSMUSG00000118555::chr7:111119594-111120094(-))
            tmp_dct[k] = seq
        ## add feat sequences dct 
        ## to main dct
        fasdct[feat] = tmp_dct
    
    pickle.dump(fasdct, open(outpkl, "wb" ) )
    return fasdct

#### PREPROCESS
# %% MAIN - INTERACTIVE
# ## expression data
# non_exprs_idxs  = [0,1] ## indexes for columns other than exprssion data i.e. gene info, etc.
# data_df         = data_reader(DATA_FL)
# data_trfd       = process_exprs_data(data_df, non_exprs_idxs, id_col = 0, method = 'log')

## sequence data
# bed         = clean_bed(GENE_BED_FL, remove_scaffolds=True)
# bed_uniq    = gene_level_bed(bed, feat = 'promoter')
# prom_fasta  = fetch_promoters(bed_uniq, CHR_SIZE_FL, GENOME_ASSM, flank = 500)
# fasta_lst   = [prom_fasta, ]
# feats_lst   = ['promoter', ]
# fasdct      = read_seq_features(fasta_lst, feats_lst)

# %% DEV


# %%
# ense_ids    = set(data_trfd['Gene.ID'])
# fasdf       = pd.DataFrame(fasdct['promoter'].keys())
# biomart_df  = pd.read_csv('biomart_anno_5_utr.tsv', sep = "\t")

# %% 
# fas_heads   = set(fasdct['promoter'].keys())
# biomart_ids = set(biomart_df['ensembl_gene_id'])
# lst = []
# for i in ense_ids:
#     if i not in biomart_ids:
#         lst.append(i)

# %%
# df1 = pd.DataFrame(data_trfd['Gene.ID'])
# biomart_join_df = biomart_df.merge(df1, how= 'outer', left_on = 'ensembl_gene_id', 
#                                     right_on = 'Gene.ID', indicator = True, suffixes = ["_bm", "_exprs"])
# biomart_join_df.to_csv('biomart_merged_exprs.tsv', sep = "\t")

# %% TEST
# fig = exprs_tf.plot.box(figsize=(20,8), rot = 90).get_figure()
# fig.savefig('log_norm.pdf')

# %% MAIN 
def main():
    ## expression data
    # non_exprs_idxs = [0,1] ## indexes for columns other than exprssion data i.e. gene info, etc.
    # data_df        = data_reader(DATA_FL)
    # data_trfd      = process_exprs_data(data_df, non_exprs_idxs, id_col = 0, method = 'log')

    ## seq data 
    ## read, clean and extract bed
    bed         = clean_bed(GENE_BED_FL, remove_scaffolds=True)
    bed_uniq    = gene_level_bed(bed, feat = 'promoter')
    prom_fasta  = fetch_promoters(bed_uniq, CHR_SIZE_FL, GENOME_ASSM, flank = 500)
    fasta_lst   = [prom_fasta, ]
    feats_lst   = ['promoter', ]
    fasdct      = read_seq_features(fasta_lst, feats_lst)

    return None

# %%
if __name__ == "__main__":
    main()
    pass


# %% CHANGELOG
## v01 [12/31/2020]
## added functions to read imputed expression data
## check for missin values
## log transform
## and write as dataframe and dict 
## plot ransformed data

## v02 [01/15/2021]
## reverted output to tpm counts and not transformed data
## removed empty_cells columns after checing for empty cols in 'process_exprs_data'

## v03 [01/24/2021]
## added functions to fetch promoter seqeunces guven bedfiles
## and write the dictionary for seqeunce based features (most of the functions moved to fetchpromoters.py)

## v04 [02/02/2021]
## added function to segment DNA seqeunces before writting to dict; segemenattion is required 
##      to generate the embeddings




# %%
#### TO DO
## 1. How to manage rows with no values, or value for some columns - check article?
