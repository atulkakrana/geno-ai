# Diseases Gene Discovery Using Artificial Neural Networks
<br/>

## I. Gene Expression Datasets (GED)
Gene expression datasets for different Illumina platforms is collected from multiple
organs, development stages and perturbation experiments.

It would be useful to get 1K data set i.e. 1024 different data (i.e. columns) for main
human organs/tissues (stages) such as eye, kidney, craniofacial, etc.  

Iy may also be useful to use datsets from different models to account for inter-species differences


### Datasets/Papers
1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6658352/ [Bulk RNA-seq 7 tissues multiple stages]
    * E-MTAB-6769 (chicken), 
    * E-MTAB-6782 (rabbit)
    * E-MTAB-6798 (mouse)
    * E-MTAB-6811 (rat)
    * E-MTAB-6813 (rhesus)
    * E-MTAB-6814 (human)
    * E-MTAB-6833 (opossum)


2. https://www.ebi.ac.uk/gxa/sc/experiments/E-ENAD-15/results/tsne [Single-cell multiple tissues]



<br/>
<br/>
<br/>

## II. Processing GED For Gene Vectors
GED are processed for normalized expressions counts. These are likely normalized that 
sum of normalized expression is scaled between 0 and 1. These represent as gene expression
vectors.

### 1. Compute Expression Counts

<br/>

### 2. Impute Missing Expression Counts

**Basic Imputation**: Replace all missing FPKM counts to 0.    
**KNN**: https://www.bioconductor.org/packages/devel/bioc/manuals/impute/man/impute.pdf    
**GAN**: https://www.biorxiv.org/content/10.1101/2020.06.09.141689v1    
**AE**:  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7144625/    


<br/>

<br/>      
<br/>
<br/>

## III. Generate Training Set
Labels could be fecthed for genes from [DisGeNet site](https://www.disgenet.org/). Model as multi-label
classification experiment by providing multiple diseases labels to each gene (using some cutoff). 

### Features
1. Gene expression
2. Kmer-embeddings from promotor regions
3. Keywords or sentiments associated with genes