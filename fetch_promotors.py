## code and functions for 
## calling Ensembl APIs
## see approach: https://www.biostars.org/p/182727/


# %% ENVIRONMENT
import os, sys
from os.path import expanduser
HOME = expanduser("~")
# sys.path.insert(0, f'{HOME}/genome')

# %% IMPORTS
import pandas as pd
import subprocess
from subprocess import PIPE

# %% TOOLS

# %% SETTINGS
UCSC_GENE_TABLE = f"{HOME}/0.work/genomes/hgTables_gene.txt"
CHR_LENGTHS     = f"{HOME}/0.work/genomes/grcm38.txt"
ASSEMBLY        = f"{HOME}/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.mod.fa"

## HELPERS
def add_flanks(gene_table, chr_lengths, flank = 500):
    '''
    add flanking regions to gene coords table
    from the UCSC browser;

    output: each line is duplicated to include flanking regions 
    from the other strand; we will filter these later
    '''

    ## outputs
    outfile = "%s_flanked.bed" % (gene_table.rpartition(".")[0])

    ## inputs
    flank   = str(flank)

    ## run bedtools    
    opts = ['bedtools', 'flank','-l', flank, '-r', flank, '-i', gene_table, '-g', chr_lengths]
    
    print("%s" % (" ".join(str(i) for i in opts)))
    res = subprocess.run(opts, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    
    ## check output and write results
    if res.returncode == 0:
        print("Process finished")
        print(f"see outfile:{outfile}")
        fhout = open(outfile, 'w')
        fhout.write("%s" % (res.stdout))
        fhout.close()
        pass
    else:
        print("Something went wrong with bedtools flanks generator")
        print(f"STD ERROR:{res.stderr}")
        sys.exit()

    return outfile

def filter_flanked(bedfile, gene_table):
    '''
    bedtools when flanking duplicates gene entry so include 
    coords for flanks for both left and right; here we retain 
    right coords for + strand and left coords for negative strand
    '''
    ## outfile
    outfile = "%s_flanked_uniq.bed" % (gene_table.rpartition(".")[0])
    fhout   = open(outfile,'w')
    outdct  = {}

    ## inputs
    df   = pd.read_csv(bedfile, sep = "\t", header=None)
    grp  = df.groupby(df.iloc[:,3])
    keys = grp.groups.keys()

    ## iterate over the entries
    ## and select one entry per gene
    ## based on the strand
    for k,v in list(grp.groups.items()):
        idxs    = list(v)
        ents    = [list( df.iloc[x,:] ) for x in idxs]
        strand  = ents[0][5] ## pick starnd from first entry
        # print(f"\nKey:{k} | idxs:{idxs}")

        ## sanity check
        if len(idxs) != 2:
            print(f"Two entries expected per genes")
            print(f"entries:{len(idxs)} - proceed carefully")
            print(f"\nKey:{k} | idxs:{v}")
            print(f"ENTS:{ents}")
            ent = ents[0]
        else:
            ## pick left or right 
            ## flank based on strand
            if strand == "+":
                ent = ents[0]
            else:
                ent = ents[1]
                
            
            ## add to output dict
            ## and write to file
            outdct[k] = ent
            fhout.write("%s\n" % ("\t".join(str(i) for i in ent)))
    
    fhout.close()
    print(f"\nInput:{len(df.index)} | output:{len(outdct)}")
    print(f"See outfile:{outfile}")
    return df, outdct

def extract_seqs(gene_table, assembly):
    '''
    use bedtools and coordinates from bed file to extract the sequences;

    flip the seqeunces on negative strand??
    '''

    ## outfile
    outfile = "%s_flanked_uniq.fasta" % (gene_table.rpartition(".")[0])
    fhout   = open(outfile,'w')

    ## inputs
    inpfile = "%s_flanked_uniq.bed" % (gene_table.rpartition(".")[0])
    print(f"Coords file:{inpfile}")

    ## Use bedtools to 
    ## extract seqeunces
    ## '-s' switch to use strand
    ## informarion and reverse
    ## complement for negative strand
    opts    = ['bedtools','getfasta', '-fi', assembly, '-bed', inpfile, '-fo', outfile, '-s', '-name']
    print("%s" % (" ".join(str(i) for i in opts)))
    res = subprocess.run(opts, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    ## check output and write results
    if res.returncode == 0:
        # print("Process finished")
        print(f"Errors (if any):{res.stderr}")
        print(f"see outfile:{outfile}")
        pass
    else:
        print("Something went wrong with bedtools flanks generator")
        print(f"STD ERROR:{res.stderr}")
        sys.exit()
    
    return outfile

# %% MAIN - Interactive
# flanked_bed      = add_flanks(UCSC_GENE_TABLE, CHR_LENGTHS, flank = 500)
# df, flanked_dct  = filter_flanked(flanked_bed, UCSC_GENE_TABLE)
_ = extract_seqs(UCSC_GENE_TABLE, ASSEMBLY)


# %% DEV

# %%
outdct = {}
for k,v in list(grp.groups.items()):
    idxs    = list(v)
    ents    = [df.iloc[x,:] for x in idxs]
    strand  = ents[0][5] ## pick starnd from first entry
    # print(f"\nKey:{k} | idxs:{idxs}")

    ## sanity check
    if len(idxs) != 2:
        print(f"Two entries expected per genes")
        print(f"entries:{len(idxs)} - proceed carefully")
        print(f"\nKey:{k} | idxs:{v}")
        print(f"ENTS:{ents}")
        ent = ents[0]
    else:
        ## pick left or right 
        ## flank based on strand
        if strand == "+":
            ent = ents[0]
        else:
            ent = ents[1]
        
        ## add to output dict
        outdct[k] = ent

print(f"Input:{len(df.index)} | output:{len(outdct)}")

# %% TEST


# %% MAIN
def main():
    return None

# %%
if __name__ == "__main__":
    main()
    pass


# %% CHANGELOG
## added '-s' parameter to getfasta command to reverse complement negative starnd coords (tested okay)
## added '-name' to getfasta coomand to include gene name

# %% NOTES
## Promotors can also be downloaded manually from UCSC table browser
##  choose genome {Mouse} assembly {GRCm38/mm10}, default track {GENCODE VM23},
##  table {knownGene}, output format {seqeunce}, and click on download
##  On next page, select {genomic} then choose promotors (untick all other seq types)
##  Note: for reverse stand, the UCSC automatically reverse complements the promoter seqeunce
##  which is the right thing to do.