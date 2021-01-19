## code to download promotors from biomart

#### IMPORTS #############
library("biomaRt")
library("GenomicRanges")


### USER SETTINGS ########



#### EXPLORE BIOMART #####
## databases
marts = listMarts() ## see different marts
marts[:10,]

## datasets
datasets = listDatasets(ensembl)
datasets[1:150,]

## filters
filters = listFilters(ensembl)
filters[1:100,1]

## attributes
attributes = listAttributes(ensembl)
attributes[1:100,1]

#### SELECT DATA #########
mart    = "ensembl"
dataset = "mmusculus_gene_ensembl" ## from datasets list above; grc38.p6
ensembl = useDataset(mart=ensembl, dataset=dataset)

#### GENERATE QUERY
qids    = c('ENSMUSG00000000001','ENSMUSG00000000028') ## test ensembl ids
attrbs  = c('ensembl_gene_id', 'hgnc_symbol', 'external_gene_name',
            'chromosome_name','5_utr_start','strand', 'source')
filts   = c('ensembl_gene_id')

annots  = getBM(attributes  = attrbs, 
                filters     = filts, 
                values      = qids, 
                mart        = ensembl)
annots

#### FETCH SEQUENCES
mart    = useMart("ensembl", dataset="mmusculus_gene_ensembl")
for (i in 1:length(annots)-1) {
  ense = annots$ensembl_gene_id[[i]]
  name = annots$hgnc_symbol[[i]]
  chr  = annots$chromosome_name[[i]]
  print(ense)
  print(name)
  print(chr)
  print(".")

  ## Fetch Seq
  seq     = getSequence(id      = ense,
                        type    = 'ensembl_gene_id',
                        seqType = 'gene_flank',
                        upstream= 300,
                        mart    = mart)
  show(seq)
}

