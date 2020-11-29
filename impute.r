## IMPUTE THE GENE EXPRESSION DATA
## atulkakrana@gmail.com

## ENVIRONMENT
Sys.setenv(
  "AWS_ACCESS_KEY_ID"     = "AKIAY4KB5K45YWZTFZOJ",
  "AWS_SECRET_ACCESS_KEY" = "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",
  "AWS_DEFAULT_REGION"    = "us-east-2"
)

## IMPORTS
require(impute)
require(aws.s3)

## READ DATA
s3read_using(FUN = read.csv2, bucket = "lachke-lab-data/work/0.geno-ai/data/rna-seq/E-MTAB-6798/", 
                              object = "E-MTAB-6798-query-results.fpkms.tsv")


df_main_fpkm = read.delim2("/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.tsv", 
                              header = TRUE, sep = "\t")
nrow(df_main_fpkm);ncol(df_main_fpkm)
# df_main_fpkm[df_main_fpkm == ""] <- 0
df_main_fpkm[1:10, 1: 8]

colnames(df_main_fpkm)


## Extract expression matrix
mt_main_fpkm = as.matrix(sapply(df_main_fpkm[,3:96], as.numeric))
nrow(mt_main_fpkm); ncol(mt_main_fpkm)
mt_main_fpkm[1:5,1:5, drop = FALSE]


## Impute; output is a list
mt_impute_lst  = impute.knn(mt_main_fpkm ,k = 10, rowmax = 0.5, colmax = 0.8, maxp = 1500, rng.seed=362436069)
names(mt_impute_lst) ## Check list elements
mt_impute_fpkm = mt_impute_lst$data
nrow(mt_impute_fpkm ); ncol(mt_impute_fpkm )
mt_impute_fpkm[1:5,1:8, drop = FALSE]


## Make DF and write
df_impute_fpkm = as.data.frame(mt_impute_fpkm)
df_impute_fpkm[1:5, 1:5]

gene.id   = df_main_fpkm['Gene.ID']
gene.name = df_main_fpkm['Gene.Name']

df_impute_final = cbind(gene.id, gene.name, df_impute_fpkm)
colnames(df_impute_final)[1] = 'Gene.ID'
colnames(df_impute_final)[2] = 'Gene.Name'

## Write Dataframe
write.table(df_impute_final, "/home/atul/0.work/0.dl/data/rna-seq/e-mtab-6798/E-MTAB-6798-query-results.fpkms.impute.tsv", 
                             sep = "\t", row.names = FALSE)

