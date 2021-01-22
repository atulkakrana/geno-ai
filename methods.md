# Methods

## Preprocessing


## Promoter Sequences
Downloaded gene coodinates from UCSC table browser (assembly Grcm38/mm10) in BED format. These were flanked to 500 nucleotides using Bedtools `flank` feature. The seqeunce for flanked coodinates is extracted using Bedtools `getfasta` feature with `-s` switch to ensure the correct orientation/direction for promoter seqeunces on the negative strand. 