# Diseases Gene Discovery Using Artificial Neural Networks
<br/>

## I. Gene Expression Datasets (GED)
Gene expression datasets for different Illumina platforms is collected from multiple
organs, development stages and perturbation experiments.

It would be useful to get 1K data set i.e. 1024 different data (i.e. columns) for main
human organs/tissues (stages) such as eye, kidney, craniofacial, etc.  


### Datasets/Papers
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6658352/

<br/>

## II. Processing GED For Gene Vectors
GED are processed for normalized expressions counts. These are likely normalized that 
sum of normalized expression is scaled between 0 and 1. These represent as gene expression
vectors.

<br/>      

## III. Generate Training Set
Labels could be fecthed for genes from [DisGeNet site](https://www.disgenet.org/). Model as multi-label
classification experiment by providing multiple diseases labels to each gene (using some cutoff). 

### Features
1. Gene expression
2. Kmer-embeddings from promotor regions
3. Keywords or sentiments associated with genes