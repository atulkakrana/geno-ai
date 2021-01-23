## this code computes embeddings 
## for DNA seqeunces

# %% ENVIRONMENT
import os, sys
from os.path import expanduser
HOME = expanduser("~")


# %% IMPORTS
import sys
import pathlib
import datetime
import fasttext


# %% SETTINGS
DATA_DIR = f"{HOME}/0.work/genomes"
DATA_FLS = ['hgTables_5UTR.fa', 'hgTables_3UTR.fa', 'hgTables_500up.fa', 'hgTables_300down.fa', 'hgTables_cds.fa']


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

    print(f"\nRead file:{fas} | Seqeunces:{len(fasdct)}")
    return fasdct

def process_genome_data(datadir, datafls):
    '''
    reads llist of fasta files; extacts seq and 
    writes seq as newline seprated fasta seqeunce (one in each line)
    '''

    ## output
    reslst = []

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
    _ = fasta_writer(reslst, datadir)

    print(f"Processed:{acount} fasta files")
    print(f"Total genomic features (i.e. seqeunces):{len(reslst)}")
    return reslst

def fasta_writer(faslst, datadir):
    '''
    writes two element list (name, seq) to fasta
    '''
    ## output
    outfile = "%s/all_features_seq.fa" % (datadir)
    fhout = open(outfile, 'w')

    ## write to file
    acount = 0
    for idx, aseq in enumerate(faslst):
        fhout.write("%s\n\n" % (aseq))
        acount +=1
    fhout.close()
    print(f"\nWrote {acount} seqs to file:{outfile}")
    return None

def train_word_vec_model(feature_file):
    '''
    here we train word vectors models
    https://fasttext.cc/docs/en/unsupervised-tutorial.html
    https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/
    '''

    ## output
    timestamp   = datetime.datetime.now().strftime("%m_%d_%H_%M")
    model_file  = "%s_%s.bin" % (feature_file.rpartition(".")[0], timestamp)
    model_mem   = "%s_word_vec_%s.txt" %  (feature_file.rpartition(".")[0], timestamp)
    fhout       = open(model_mem, 'w')

    ## sanity check and 
    ## train word vectors
    minn_val    = 2    ## min subword size
    maxn_val    = 6    ## max subword size
    dimn_val    = 128  ## embedding dimensions
    epochs      = 20   ## epochs; default:5
    learn_rate  = 0.01 ## default: 0.5 [0.01,1]
    path        =  pathlib.Path(feature_file)

    print(f"Training fastText words vectors on seq file:{feature_file}")
    print(f"featurefile:{feature_file}\nminn:{minn_val}\nmaxn:{maxn_val}\ndim:{dimn_val}\nepoch:{epochs}\nlearning_rate:{learn_rate}\n")
    if path.is_file():
        model = fasttext.train_unsupervised(feature_file, minn=minn_val, maxn=maxn_val, dim=dimn_val, epoch = epochs, lr = learn_rate)
        model.save_model(model_file)
        fhout.write(f"featurefile:{feature_file}\nminn:{minn_val}\nmaxn:{maxn_val}\ndim:{dimn_val}\nepoch:{epochs}\nlearning_rate:{learn_rate}\n")
    else:
        raise FileNotFoundError(f"{feature_file} not found")
    
    fhout.close()
    print(f"Trained word vector model:{model_file}")
    return model

# %% MAIN
# out_fasta       = process_genome_data(DATA_DIR, DATA_FLS)
# test_file       = "/home/atul/0.work/genomes/all_features_seq.fa"
test_file       = "/home/atul/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.forfasttext.fa"
wordvec_model   = train_word_vec_model(test_file)

# %% DEV


# %% TEST
# model = fasttext.train_unsupervised('embeddings/data/fil9', "cbow")
# # %%
# model.get_nearest_neighbors('dragon')

# %% MAIN
def main():
    out_fasta = process_genome_data(DATA_DIR, DATA_FLS)

    return None

# %% 
if __name__ == "__main__":
    main()
    pass


#### CHANGELOG
## v01 [01/21/2021]



