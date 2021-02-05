#!/usr/bin/env python

## This code processes the 
## DNA sequence data and 
## trains seq embedidngs 
## model for vectorizing
## the DNA seqeunces (whole
## genome or FASTA files)

# %% ENVIRONMENT
import os, sys
from os.path import expanduser
HOME = expanduser("~")


# %% IMPORTS
import sys
import pathlib
import datetime
import fasttext
import pickle as pkl
from dlcore import gen_seq_embeddings


# %% SETTINGS
DATA_DIR = f"{HOME}/0.work/genomes"
DATA_FLS = ['hgTables_5UTR.fa', 'hgTables_3UTR.fa', 'hgTables_500up.fa', 'hgTables_300down.fa', 'hgTables_cds.fa'] ## Train Model Genomic Features
# DATA_FLS = ['Mus_musculus.GRCm38.68.dna.toplevel.mod.clean.fa'] ## Train Model On Whole Genome

# %% FUNCTIONS
def fasta_reader(fas):
    '''
    read fasta file to list
    '''

    fhin        = open(fas, 'r')
    aread       = fhin.read()
    faslst      = aread.split('>')
    fhin.close()

    fasdct      = {}    ## Store FASTA as dict
    acount      = 0     ## count the number of entries
    empty_count = 0
    for i in faslst[1:]:
        ent     = i.split('\n')
        aname   = ent[0].split()[0].strip()       
        seq     = ''.join(x.strip() for x in ent[1:]) ## Sequence in multiple lines
        alen    = len(seq)
        fasdct[aname] = seq

    print(f"Read FASTA file:{fas} | Sequences:{len(fasdct)}")
    return fasdct

def fasta_writer(fas_dct, fas_out):
    '''
    writes fasta dict to fasta file;
    the output filename is provided as an input
    '''

    ## outfile
    fhout = open(fas_out, 'w')

    ## iterate and write
    acount = 0
    for head, seq in fas_dct.items():
        fhout.write(">%s\n%s\n" % (head, seq))
        acount+=1
    fhout.close()

    print(f"Written FASTA file with {acount} | Sequences:{fas_out}")
    return fas_out

def feats_writer(faslst, feats_out):
    '''
    writes two element list (name, seq) to fasta
    '''
    ## output
    # outfile = "%s/all_features_seq.fa" % (datadir)
    fhout = open(feats_out, 'w')

    ## write to file
    acount = 0
    for idx, aseq in enumerate(faslst):
        fhout.write("%s\n\n" % (aseq))
        acount +=1
    fhout.close()
    print(f"\nWrote {acount} seqs to file:{feats_out}")
    return feats_out

def combine_fasta(datadir, datafls):
    '''
    reads list of fasta files; extacts seq and 
    writes seq as newline seprated fasta seqeunce (one in each line)
    '''
    print(f"\nFn: Combine FASTA Seqeunces")

    ## output
    reslst      = []

    if len(datafls) > 1:
        feats_out   = os.path.join(datadir, "genomic_features-train.fa")
    else:
        afile = datafls[0]
        feats_out   = os.path.join(datadir, "%s-train.fa" % afile.rpartition(".")[0] )

    acount = 0
    for fas in datafls:
        path   = os.path.join(datadir, fas)
        fasdct = fasta_reader(path)
        acount+=1

        ## add sequences to 
        ## master list
        tmplst = list(fasdct.values())
        print(f"Cached file:{path} | seqs:{len(tmplst)}")
        reslst.extend(tmplst)

    ## write to file
    fas_dct = {idx:seq for idx,seq in enumerate(reslst)}
    _ = fasta_writer(fas_dct, feats_out)

    print(f"Processed:{acount} fasta files")
    print(f"Total genomic features (i.e. seqeunces):{len(reslst)}")
    return reslst, feats_out

def process_seqs(fas_in, method = "ok", out_format = "fasta"):
    '''
    Perform FASTA segmentation or processing for 
    embeddings.

    Available Methods:
    "ok": Overlapping, sliding, K-mer given a width and a stride input

    Available file outputs (i.e. out_format):
    "fasta" = fasta file (for pre-processing fasta before embeddings)
    "feats" = text file with just line separated seqeunces (for training mebddings model)
    '''
    print(f"\n#### Fn: Seqeunce Segementator ########")

    ## outputs
    fas_dct     = {}
    fas_out     = "%s-segmented.fas"    % (fas_in.rpartition(".")[0])
    feat_out    = "%s-segmented.feats"  % (fas_in.rpartition(".")[0])

    ## inputs
    fas_dct = fasta_reader(fas_in)

    ## iterate and segment
    if method == "ok":
        for head, seq in fas_dct.items():
            kmer_lst = overlap_k(seq, kwidth =3, stride = 1)
            kmer_seq = " ".join(str(i) for i in kmer_lst)
            fas_dct[head] = kmer_seq
    else:
        print(f"Method not implemented:{method} - exiting")
        sys.exit()

    ## write fasta file
    if out_format   == "fasta":
        outfile     = fasta_writer(fas_dct, fas_out)
        print(f"Sequence segementaion complete")
    elif out_format == "feats":
        feat_lst    = fas_dct.values()
        outfile     = feats_writer(feat_lst, feat_out)
        print(f"Sequence segementaion complete")
    else:
        print(f"The `out_format` value:{out_format} not recofnized - exiting")
        sys.exit()

    print(f"See outfile:{outfile}")
    return fas_dct, outfile

def overlap_k(seq, kwidth =3, stride =1):
    '''
    sliding window for overlapping k-mers of given size after given strides
    '''

    ## output
    kmer_lst = []

    ## size
    owidth = int( ( ( len(seq.strip())-kwidth )/stride ) +1 )
    # print(f"\nSeq Len:{len(seq)} | Output width:{owidth}")

    idx = 0
    for n in range(owidth):
        s       = idx
        kmer    = seq[s:s+kwidth]
        kmer_lst.append(kmer)
        idx     += stride

    # print(f"Kmers snippet:{kmer_lst[:50]}")
    # print(f"Seq length: {len(seq)} | Total Kmers:{len(kmer_lst)}")
    return kmer_lst

def fasta_to_embed(infas, model_fl):
    '''
    this is the function to convert fasta file 
    to a dict of embeddings
    '''

    ## output
    outembed    = "%s-embed.p" % (infas.rpartition(".")[0])
    embed_dct   = {}

    ## segment fasta
    _, feats_fl = process_seqs(infas, method = "ok", out_format = "fasta")
    feats_dct   = fasta_reader(feats_fl)

    ## embed
    embed_dct   = gen_seq_embeddings(feats_dct, model_fl)

    print(f"Genes in input FASTA:{len(feats_dct)} | Genes in output embeddings:{len(embed_dct)}")
    pkl.dump(embed_dct, open(outembed, "wb"))
    return embed_dct

def train_word_vec_model(feature_file, model_type = "cbow"):
    '''
    here we train word vectors models
    https://fasttext.cc/docs/en/unsupervised-tutorial.html
    https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/
    
    Available Models Types:
    "skipgram"  = Skip-Gram
    "cbow"      = continuous bag of words
    '''

    ## output
    timestamp   = datetime.datetime.now().strftime("%m_%d_%H_%M")
    model_file  = "%s_%s.bin" % (feature_file.rpartition(".")[0], timestamp)
    model_mem   = "%s_word_vec_%s.txt" %  (feature_file.rpartition(".")[0], timestamp)
    fhout       = open(model_mem, 'w')

    ## sanity check and 
    ## train word vectors
    minn_val    = 3    ## min subword size
    maxn_val    = 6    ## max subword size
    dimn_val    = 128  ## embedding dimensions
    epochs      = 5   ## epochs; default:5
    learn_rate  = 0.05 ## default: 0.5 [0.01,1]
    path        =  pathlib.Path(feature_file)

    print(f"Training fastText words vectors on seq file:{feature_file}")
    print(f"featurefile:{feature_file}\nminn:{minn_val}\nmaxn:{maxn_val}\ndim:{dimn_val}\nepoch:{epochs}\nlearning_rate:{learn_rate}\n")
    if path.is_file():
        if model_type == "skipgram":
            print(f"Training `SKIPGRAM` model")
            model = fasttext.train_unsupervised(feature_file, "skipgram", minn=minn_val, maxn=maxn_val, dim=dimn_val, epoch = epochs, lr = learn_rate)
        elif model_type=="cbow":
            print(f"Training `CBOW` model")
            model = fasttext.train_unsupervised(feature_file, "cbow", minn=minn_val, maxn=maxn_val, dim=dimn_val, epoch = epochs, lr = learn_rate)
        else:
            model = None
            print(f"Check `model_type` parameter:{model_type} - exiting")
            sys.exit()
        
        ## write the models and config
        model.save_model(model_file)
        fhout.write(f"featurefile:{feature_file}\ntype:{model_type}\nminn:{minn_val}\nmaxn:{maxn_val}\ndim:{dimn_val}\nepoch:{epochs}\nlearning_rate:{learn_rate}\n")
    
    else:
        raise FileNotFoundError(f"{feature_file} not found")
    
    fhout.close()
    print(f"Trained word vector model:{model_file}")
    return model

# %% MAIN
## Preapre Feats Seqeunce
# _, comb_fasta   = combine_fasta(DATA_DIR, DATA_FLS)
# _, train_fasta  = process_seqs(comb_fasta, out_format="feats")

## Train Model
# train_fasta           = "/home/atul/0.work/genomes/all_features_seq-segmented.test.feats"
# train_fasta           = "/home/atul/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.forfasttext.fa"
# wordvec_model   = train_word_vec_model(train_fasta)

# %% DEV
# model = fasttext.load_model("/home/atul/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.forfasttext_01_23_13_11.bin")

# %%DEV
# process_seqs(fas_in = "test.fa", method = "ok")

# %% TEST
# vect = model.get_word_vector('ATGCGC')
# model.get_nearest_neighbors('TTGCGC')
# vect.shape

# %% MAIN
def main():
    _, comb_fasta   = combine_fasta(DATA_DIR, DATA_FLS)
    _, train_file   = process_seqs(comb_fasta, out_format="feats")
    # train_file       = "/home/atul/0.work/genomes/all_features_seq.fa"
    # train_file       = "/home/atul/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.mod.clean-train-segmented.feats"
    wordvec_model   = train_word_vec_model(train_file)

    return None

# %% 
if __name__ == "__main__":
    main()
    pass


#### CHANGELOG
## v01 [01/21/2021]
## wrote functions to read/write FASTA files and feature files for training
## wrote functions for training FASTTEXT on DNA seqeunces

## v01 -> v02 [02/01/2021]
## added function to segment DNA seqeunces
## reused `process_genome_data` to `combine_fasta` for downstream segemetation



