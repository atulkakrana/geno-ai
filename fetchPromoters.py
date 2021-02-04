## code and functions for 
## calling Ensembl APIs
## see approach: https://www.biostars.org/p/182727/


# %% ENVIRONMENT
import os, sys
from os.path import expanduser
HOME = expanduser("~")
# sys.path.insert(0, f'{HOME}/genome')

# %% IMPORTS
import subprocess
import numpy as np
import pandas as pd
from subprocess import PIPE
from seq_embeddings import fasta_reader

# %% TOOLS

# %% SETTINGS
UCSC_GENE_TABLE = f"{HOME}/0.work/genomes/hgTables_gene.txt"
CHR_LENGTHS     = f"{HOME}/0.work/genomes/grcm38.txt"
ASSEMBLY        = f"{HOME}/0.work/genomes/Mus_musculus.GRCm38.68.dna.toplevel.mod.fa"

# %% HELPERS
def clean_bed(bed, remove_scaffolds = False):
    '''
    cleans bed file for better comaptibility with genomes
    remove_scaffold: will remove seqeunces from scaffolds and contigs
    '''
    print("\n#### Fn: Clean BED File #######")

    ## lists
    exceptions = ['X', 'Y', 'MT', 'M']

    ## output
    outfile = "%s.clean.bed" % (bed.rpartition(".")[0])
    fhout = open(outfile, 'w')

    ## input
    fhin = open(bed, 'r')
    read = fhin.readlines()
    fhin.close()

    ## cleanup
    acount = 0
    bcount = 0
    for i in read:
        # print(f"Ent:{i}")
        ent     = i.strip("\n").split("\t")
        chr     = ent[0]
        abool   = False ## to track scaffold entries
        acount  += 1

        tmp = chr.replace('chr', "")
        if tmp.isnumeric():
            chr = "chr"+str(tmp)
        elif tmp in exceptions:
            chr = "chr"+str(tmp)
        else:
            chr = "chr_"+str(tmp)
            abool = True
        
        ent_c    = list(ent)
        ent_c[0] = chr

        if (remove_scaffolds == True) and (abool == True):
            print(f"Removing:{ent}")
            pass
        else:
            fhout.write("%s\n" % ("\t".join(str(i) for i in ent_c)) )
            bcount+=1

    fhout.close()
    print(f"BED entries read:{acount}")
    print(f"BED entries (cleaned) and written:{bcount}")
    print(f"Updated file is :{outfile}")
    return outfile

def gene_level_bed(bed, feat = 'prom'):
    '''
    the bed files may have multiple entries for single gene i.e.
    different reported 5' UTR coordinates for same genes; here
    we ensure that there is just one entry per gene level
    '''
    print("\n#### Fn: Collapse To Gene Level #######")

    ## output
    outfile = "%s.gene-level.bed" % (bed.rpartition(".")[0])
    fhout = open(outfile, 'w')

    ## inputs
    df   = pd.read_csv(bed, sep = "\t", header=None)
    grp  = df.groupby(df.iloc[:,3])
    keys = grp.groups.keys()

    ## iterate over the entries
    ## and select one entry per gene
    ## based on the strand
    uniq_lst = []
    if (feat == 'promoter') or (feat == '5utr'):
        for k,v in list(grp.groups.items()):
            idxs    = list(v)
            ents    = [list( df.iloc[x,:] ) for x in idxs]
            lft_lst = [int(x[1]) for x in ents]
            rgt_lst = [int(x[2]) for x in ents]
            strand  = ents[0][5] ## pick strand from first entry
            # print(f"\nKey:{k} | idxs:{idxs}")
            # print(f"Left coords:{lft_lst}")
            # print(f"Right coords:{rgt_lst}")

            ## select ent based on strand
            ## information
            if (strand == "+") or (strand == 1):
                arr  = np.array(lft_lst)
                idx  = np.argmin(arr)
                uniq = ents[idx]
                uniq_lst.append(uniq)
            elif (strand == "-") or (strand == -1):
                arr  = np.array(rgt_lst)
                idx  = np.argmax(arr)
                uniq = ents[idx]
                uniq_lst.append(uniq)
            else:
                print(f"Strand is not known:{strand} - exiting")
                sys.exit(0)


    else:
        print(f"No strategy implemnted to uniq this feature:{feat} - exiting")
        sys.exit()
    
    ## write uniq entries to a file
    for ent in uniq_lst:
        fhout.write("%s\n" % ("\t".join(str(i) for i in ent)))

    print(f"ntries in input BED file:{len(df.index)}")
    print(f"Entries collapsed to gene level:{len(uniq_lst)}")
    print(f"Updated file is :{outfile}")
    return outfile


# %% FUNCTIONS
def add_flanks(gene_table, chr_lengths, flank = 500):
    '''
    add flanking regions to gene coords table
    from the UCSC browser;

    output: each line is duplicated to include flanking regions 
    from the other strand; we will filter these later
    '''
    print(f"\n#### Fn: Flank Coordinates #####")

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
    
    print(f"Output file is :{outfile}")
    return outfile

def filter_flanked(bedfile):
    '''
    bedtools when flanking duplicates gene entry so include 
    coords for flanks for both left and right; here we retain 
    right coords for + strand and left coords for negative strand
    '''
    print(f"\n#### Fn: Select Flank Coords ########")
    
    ## outfile
    outfile = "%s_uniq.bed" % (bedfile.rpartition(".")[0])
    fhout   = open(outfile,'w')
    outdct  = {}

    ## inputs
    df      = pd.read_csv(bedfile, sep = "\t", header=None)
    grp     = df.groupby(df.iloc[:,3])
    keys    = grp.groups.keys()

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
    print(f"Input:{len(df.index)} | output:{len(outdct)}")
    print(f"See outfile:{outfile}")
    return outdct, outfile

def extract_seqs(bedfile, assembly):
    '''
    use bedtools and coordinates from bed file to extract the sequences;

    flip the seqeunces on negative strand??
    '''
    print(f"\n#### Fn: Extract Coords Sequence ########")

    ## outfile
    outfile = "%s.fas" % (bedfile.rpartition(".")[0])
    fhout   = open(outfile,'w')

    ## inputs
    # inpfile = "%s_flanked_uniq.bed" % (gene_table.rpartition(".")[0])
    print(f"Coords file:{bedfile}")

    ## Use bedtools to 
    ## extract seqeunces
    ## '-s' switch to use strand
    ## informarion and reverse
    ## complement for negative strand
    opts    = ['bedtools','getfasta', '-fi', assembly, '-bed', bedfile, '-fo', outfile, '-s', '-name']
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
    
    print(f"See outfile:{outfile}")
    return outfile

# %% MAIN - Interactive
# flanked_bed      = add_flanks(UCSC_GENE_TABLE, CHR_LENGTHS, flank = 500)
# df, flanked_dct  = filter_flanked(flanked_bed, UCSC_GENE_TABLE)
# _ = extract_seqs(UCSC_GENE_TABLE, ASSEMBLY)


# %% DEV

# %% TEST


# %% MAIN
def main():
    flanked_bed      = add_flanks(UCSC_GENE_TABLE, CHR_LENGTHS, flank = 500)
    df, flanked_dct  = filter_flanked(flanked_bed, UCSC_GENE_TABLE)
    _ = extract_seqs(UCSC_GENE_TABLE, ASSEMBLY)
    return None

# %% PROCESS
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