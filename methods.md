# Methods

## Preprocessing


## Featurization
Data selection (binary classes), Data filteration, splits, transformation (or scaling), etc.
DisGeneNet curated diseases labels were used to generate a diseases-specific, other-diseases and unlabelled gene sets. If the genes (for other diseases i.e. neg class) was found to be associated with target-diseases in the non-curated diseases-gene set (i.e. all diseases-gene association including from NLP, etc.) then that genes was removed from negative class and move to unlabelled set. This is done to ensure that positive set is as clean as posiible.

Filters: removed genes from positive set



## Promoter Sequences
Downloaded gene coodinates from UCSC table browser (assembly Grcm38/mm10) in BED format. These were flanked to 500 nucleotides using Bedtools `flank` feature. The seqeunce for flanked coodinates is extracted using Bedtools `getfasta` feature with `-s` switch to ensure the correct orientation/direction for promoter seqeunces on the negative strand. 

Alternatively, Promotors can also be downloaded manually from UCSC table browser. Here is the method - choose genome {Mouse} assembly {GRCm38/mm10}, default track {GENCODE VM23}, table {knownGene}, output format {seqeunce}, and click on download. On the next page, select {genomic} then choose promotors (untick all other seq types). Note that for reverse stand, the UCSC automatically reverse complements the promoter seqeunce which is the right thing to do.

## Sequence Embeddings
See fast text







## Models

### CNN

### RNN