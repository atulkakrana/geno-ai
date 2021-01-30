#!/usr/bin/env python

## Stores functions required
## to generate the train set
# %% ENVIRONMENT
import os, sys
from os.path import expanduser
HOME = expanduser("~")

# %% IMPORT
from seq_embeddings import train_word_vec_model
import sys
import pickle
import fasttext
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% SETTINGS
MODEL = f"{HOME}/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.forfasttext_01_23_13_11.bin"

# %% HELPERS
def hugo_to_ense(species, aset):
    '''
    Maps hugo symbols to a Ensembl IDs
    Input: A set of gene symbols
    Output: a dict of gene symbls as keys and ense IDs as values
    '''
    ## imports
    from ensembl import gen_query_url, sym_lookup

    ## output
    map_dct   = {} ## HUGO to ENSEMBL mapping dict
    unmap_lst = [] ## HUGO that can't be mapped to ENSEMBL ID

    ## inputs
    url = gen_query_url(endpoint = 'lookup/symbol')
    inp_len = len(aset)

    acount = 0 
    for idx, asym in enumerate(aset):
        print(f"\nMapping-{idx}/{inp_len}:{asym}")
        res_json = sym_lookup(url, species, asym)
        # print(f"Response:{res_json}")
        
        if 'error' in res_json:
            text = res_json.get('error')
            print(f"Error: {text}")
            eid = None
            unmap_lst.append(asym)
        else:
            eid = res_json.get('id')
        
        if eid is not None:
            acount+=1
        else:
            pass

        print(f"Mapped Ensembl:{eid}")
        map_dct[asym] = eid

    print(f"\nTotal Symbol for mapping:{len(aset)} | mapped:{acount}")
    return map_dct, unmap_lst

def update_labs_to_ensembl(labs_dct_pkl, species):
    '''
    Input: dct with gene symbols as keys and labels (str, or list) as values 
    Converts labels key from gene symbols to ensembl
    '''

    ## imports
    import pickle
    import pandas as pd

    ## output 
    labs_dct_up = {}
    out_pkl     = "%s_ensembl.p" % (labs_dct_pkl.rpartition(".")[0])
    out_txt     = "%s_unmapped.txt" % (labs_dct_pkl.rpartition(".")[0])

    ## inputs
    labs_dct = pickle.load( open( labs_dct_pkl, "rb" ) )
    
    ## generate HUGO to ENSEMBL mappings
    map_lst  = [k.strip() for k in labs_dct.keys()]
    map_dct, unmap_lst  = hugo_to_ense(species, set(map_lst))

    ## update labels dict keys to ENSEMBL
    acount = 0
    for idx, (k,v) in enumerate(labs_dct.items()):
        eid = map_dct.get(k.strip())
        if eid is not None:
            labs_dct_up[eid] = v
        else:
            labs_dct_up[k]   = v
            acount +=1
    
    ## pickle dicts
    pickle.dump( labs_dct_up, open( out_pkl, "wb" ) )
    pd.Series(unmap_lst).to_csv(out_txt, sep = "\n", index=False, columns=None)

    print(f"Total keys:{len(labs_dct)} | Unmapped:{acount}")
    return labs_dct_up, unmap_lst

def gen_data_labels(data_pkl, labs_dct_pkl):
    '''
    the expression data has ensembl ids;
    here we use labels dct (already with ensembl keys)
    and append labels to exprs dataframe;

    the choice of dataframe is purely to keep practicing
    pandas
    '''

    ## outputs
    out_pkl     = "%s_labels.p" % (data_pkl.rpartition(".")[0])
    out_txt     = "%s_labelled.tsv" % (data_pkl.rpartition(".")[0])
    
    ## inputs
    exprs_dct   = pickle.load( open( data_pkl, "rb" ) ) ## dataframe
    labs_dct    = pickle.load( open( labs_dct_pkl, "rb" ) ) ## dict

    ## generate label series;
    ## how to handle None's?
    gids        = list(exprs_dct.keys())
    labs        = [labs_dct.get(id) for id in gids]
    labs_dct    = {id:labs_dct.get(id) for id in gids}

    ## sanity check
    if len(labs) != len(gids):
        ## labelled data may not have 
        ## information for all genes;
        ## should we skip and call these
        ## unlabelled
        print(f"The numbers keys in data doesn't match with number of IDs in lables")
        sys.exit(1)
    else:
        pass

    ## add labels to exprs data
    exprs_df    = pd.DataFrame.from_dict(exprs_dct, orient='index')
    exprs_df["Labels"] = labs

    ## pickle and write for inspection
    pickle.dump(labs_dct, open(out_pkl, "wb" ) )
    exprs_df.to_csv(out_txt, sep="\t", index = False)

    return out_pkl

def encode_labels(data_dct):
    '''
    Takes main dict of data and labels,
    and encodes labels for ML/DL

    https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/
    '''

    ## imports
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    ## inputs
    labels = data_dct['labels']
    le.fit(labels)

    ## encode labels
    labels_enc = le.transform(labels)

    return labels_enc

def gen_seq_embeddings(fasdct):
    '''
    single-level fasta dct for a feature set (i.e. gene, promoter, etc)
    key is the genename and value is seqeunce

    return a dict where seqeunce 
    key is genename and value is embedding
    '''
    print(f"\nFn: Generate Seq Embeddings")
    ## output
    embed_dct = {}

    ## inputs
    model = fasttext.load_model(MODEL)

    ## iterate and embedd
    for gene, seq in list(fasdct.items()):
        print(f"\nGene:{gene}")
        # print(f"Seq:{seq}")
        vect = model.get_word_vector(seq)
        # print(f"Vector:{vect}")
        embed_dct[gene] = vect

    print(f"Items in embed dict:{len(embed_dct)}")
    return embed_dct

# %% FUNCTIONS
def prepare_data_n_labels(dpkl, lpkl, spkl, mode='binary'):
    '''
    Process labels for ML/DL
    classifier method i.e. binary, multi-class (mclass),
    multi-label (mlabel)


    binary: retains top two labels and converts others to None;
    mclass, mlabel: retains all labels as-is and labels None/'None'

    OUTPUT: trimmed dicts with just labelled classes
    '''
    ## imports
    from collections import Counter

    ## outputs
    tdata_dct  = {}
    pdata_dct  = {}
    tdata_pkl  = "train_data.p"
    pdata_pkl  = "pred_data.p"
    
    ## inputs
    lab_dct    = pickle.load( open(lpkl, "rb" ) ) ## dict
    exp_dct    = pickle.load( open(dpkl, "rb" ) ) ## dataframe
    dna_dct    = pickle.load( open(spkl, "rb" ) ) ## dict of dicts
    pro_dct    = dna_dct['promoter'] ## dict of embedded promotors

    ## cleanup labels, if there
    ## are any artifacts
    acount = 0
    if mode == 'binary':
        vals    = list(lab_dct.values())
        # cntr    = Counter(list(vals))
        # cntr_s  = sorted(list(cntr.items()), key = lambda x: -x[1])
        # print(f"Labels Freq:{cntr_s}")

        ## visualize
        elem_cnts = [x if x is not None else 'None' for x in vals ]
        ax1       = sns.countplot(x=elem_cnts); plt.show()

        ## resticts ids to those for which
        ## all feats (expression, promoters, etc)
        ## have data - untested
        lab_ids = set(lab_dct.keys()) ## labelled IDs
        exp_ids = set(exp_dct.keys()) ## IDs with expression data
        pro_ids = set(pro_dct.keys()) ## promotors with some seq
        com_ids = set.intersection(lab_ids, exp_ids, pro_ids) ## common IDs
        print(f"Labelled Genes:{len(lab_ids)}")
        print(f"Common Genes (labelled and feats):{len(com_ids)}")

        ## collect ids for labelled data,
        ## and unlabelled data - untested
        train_ids = []
        pred_ids  = []
        for k,v in lab_dct.items():
            if k in com_ids:
                if (v is not None) and (v != 'None'):
                    train_ids.append(k)
                else:
                    pred_ids.append(k)
            else:
                # print(f"Not all feats are available for this ID:{k}")
                acount+=1
                pass
        print(f"Genes filtered out as some features were not available:{acount}")

        ## seprate labelled and unlabelled data
        print(f"Labelled instances:{len(train_ids)} | unlabelled instances:{len(pred_ids)}")
        tdata    = np.array([exp_dct[k] for k in train_ids])
        tprom    = [pro_dct[k] for k in train_ids]
        tlabs    = [lab_dct[k] for k in train_ids]
        pdata    = [exp_dct[k] for k in pred_ids]
        pprom    = [pro_dct[k] for k in pred_ids]

    else:
        ## add mclass and mlabel support
        print(f"Labeling mode:{mode} not supported")
        sys.exit(1)

    ## prepare dicts for ML/DL
    tdata_dct['exp_data']   = np.array(tdata) ## data/features and labels should be in same order
    tdata_dct['pro_data']   = np.array(tprom) ## data/features and labels should be in same order 
    tdata_dct['labels']     = tlabs ## data/features and labels should be in same order 

    pdata_dct['exp_data']   = np.array(pdata) ## data/features and labels should be in same order
    pdata_dct['pro_data']   = np.array(pprom) ## data/features and labels should be in same order 
    pdata_dct['labels']     = [None]*len(pdata) ## data/features and labels should be in same order

    ## write final train data and labels
    pickle.dump(tdata_dct, open(tdata_pkl, "wb" ) )
    pickle.dump(pdata_dct, open(pdata_pkl, "wb" ) )

    print(f"Instances in train data dct:{len(tdata)} | train labels dct:{len(tlabs)}")
    print(f"Pred dct:{len(pdata)}")
    return tdata_dct, pdata_dct

# %% MAIN - INTERACTIVE
# species      = "mouse"
# labs_dct_pkl = "labs_bin_dct.p"
# labs_dct, unmap_lst = update_labs_to_ensembl(labs_dct_pkl, species)

# LAB_PKL      = "labs_bin_dct_ensembl.p" ## could be binary labels, multi-label or multi-class
# DATA_PKL     = "data_imp_dct.p"
# labs_pkl     = gen_data_labels(DATA_PKL, LAB_PKL)

# ## generate dna embeddings
# DNA_PKL      = "dna_feats.pkl"
# dna_seqs_dct = pickle.load( open(DNA_PKL, "rb" ) )
# dna_feats_dct= {k:gen_seq_embeddings(dct) for k, dct in dna_seqs_dct.items()}
# DNA_FEATS_PKL= "dna_feats_embed.pkl"
# pickle.dump( dna_feats_dct, open(DNA_FEATS_PKL, "wb" ) )

# ## combine all feats (into one dct)
# LAB_PKL      = "data_imp_trfd_dct_labels.p" ## could be binary labels, multi-label or multi-class
# DATA_PKL     = "data_imp_dct.p"
# labs_pkl     = prepare_data_n_labels(DATA_PKL, LAB_PKL, DNA_FEATS_PKL, mode='binary')

# %% DEV

# %% TEST
# data_dct     = pickle.load( open( "train_data.p", "rb" ) )
# data_dct['exp_data'].shape


# %% MAIN
def main():
    # species      = "mouse"
    # labs_dct_pkl = "labs_bin_dct.p"
    # labs_dct, unmap_lst = update_labs_to_ensembl(labs_dct_pkl, species)

    LAB_PKL      = "labs_bin_dct_ensembl.p" ## could be binary labels, multi-label or multi-class
    # DATA_PKL     = "data_imp_trfd_dct.p"
    DATA_PKL     = "data_imp_dct.p"
    labs_pkl     = gen_data_labels(DATA_PKL, LAB_PKL)

    ## generate dna embeddings
    DNA_PKL      = "dna_feats.pkl"
    dna_seqs_dct = pickle.load( open(DNA_PKL, "rb" ) )
    dna_feats_dct= {k:gen_seq_embeddings(dct) for k, dct in dna_seqs_dct.items()}
    DNA_FEATS_PKL= "dna_feats_embed.pkl"
    pickle.dump( dna_feats_dct, open(DNA_FEATS_PKL, "wb" ) )

    ## combine all feats (into one dct)
    LAB_PKL      = "data_imp_trfd_dct_labels.p" ## could be binary labels, multi-label or multi-class
    # DATA_PKL     = "data_imp_trfd_dct.p"
    DATA_PKL     = "data_imp_dct.p"
    labs_pkl     = prepare_data_n_labels(DATA_PKL, LAB_PKL,  DNA_FEATS_PKL, mode='binary')

    return None

# %% 
if __name__ == "__main__":
    main()
    pass


# %% CHANGELOG
## v01 [12/26/2020]
## 
## v02 [01/24/2020]
## added functions to include seq embedding to train and pred data
## added check to find common genes for which all data types and labels are available
