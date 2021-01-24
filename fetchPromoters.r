## code to download promotors from biomart

#### IMPORTS #############
library("biomaRt")
library("GenomicRanges")

#### EXPLORE BIOMART #####
# ## databases
# marts = listMarts() ## see different marts
# marts
# marts_old = listEnsemblArchives()
# marts_old

## datasets
# ensembl  = useMart("ensembl")
# datasets = listDatasets(ensembl)
# datasets[1:150,]

## filters
# ensembl  = useMart("ensembl", "mmusculus_gene_ensembl")
# filters = listFilters(ensembl)
# filters[1:100,1]

## attributes
# ensembl  = useMart("ensembl", "mmusculus_gene_ensembl")
# attributes = listAttributes(ensembl)
# attributes[1:200,1]

#### SELECT DATA #########
dataset = "mmusculus_gene_ensembl" ## from datasets list above; grc38.p6
ensembl = useMart("ensembl", dataset=dataset)

#### TEST QUERY
qids    = c('ENSMUSG00000000001','ENSMUSG00000000028', 'ENSMUSG00000036095') ## test ensembl ids
attrbs  = c('chromosome_name','start_position','end_position',
            'ensembl_gene_id', 'external_gene_name','strand',
            '5_utr_start', '5_utr_end')

filts   = c('ensembl_gene_id')
annots  = getBM(attributes  = attrbs, 
                filters     = filts, 
                values      = qids, 
                mart        = ensembl)
annots
# write.table(annots, "biomart_anno_5_utr.tsv", row.names = FALSE, sep = "\t", quote = FALSE)

#### REAL QUERY FOR ALL GENE
annots  = getBM(attributes  = attrbs, 
                mart        = ensembl)
head(annots)
# write.table(annots, "biomart_anno_5_utr.tsv", row.names = FALSE, sep = "\t", quote = FALSE)


## Convert to BED format
## update strands to +/-,
## append chr to chromosomes, and
## reorder dataframe
score           = rep(0,nrow(annots)); annots['score'] = score
annots$strand[annots$strand == -1] = '-'
annots$strand[annots$strand ==  1] = '+'
annots$chromosome_name = paste("chr",annots$chromosome_name, sep="")
annots_bed      = annots[,c('chromosome_name','start_position',
                            'end_position','ensembl_gene_id', 'score', 'strand')]
head(annots_bed, 50)
write.table(annots_bed, "biomart-ensembl-anno.bed", row.names = FALSE, 
                sep = "\t", quote = FALSE, col.names = FALSE)


#### FETCH SEQUENCES
# mart    = useMart("ensembl", dataset="mmusculus_gene_ensembl")
# for (i in 1:length(annots)-1) {
#   ense = annots$ensembl_gene_id[[i]]
#   name = annots$hgnc_symbol[[i]]
#   chr  = annots$chromosome_name[[i]]
#   print(ense)
#   print(name)
#   print(chr)
#   print(".")

#   ## Fetch Seq
#   seq     = getSequence(id      = ense,
#                         type    = 'ensembl_gene_id',
#                         seqType = 'gene_flank',
#                         upstream= 300,
#                         mart    = mart)
#   show(seq)
# }

