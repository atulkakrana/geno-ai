## IMPORTS
require(Biobase)
require(GEOquery)
require(limma)
require(Glimma)
require(Mus.musculus)
require(edgeR)
require(RColorBrewer)

## Load Python Interface
library(reticulate)
use_virtualenv("ds")

os = import("os")
print(os$listdir)

# wd = getwd()
# wd

## FETCH DATA
## https://www.bioconductor.org/packages/release/bioc/vignettes/GEOquery/inst/doc/GEOquery.html
## https://wiki.bits.vib.be/index.php/Analyse_GEO2R_data_with_R_and_Bioconductor
resfolder   = "/home/atul/0.work/0.dl/data/GSE134384"
gset_1      =  getGEO("GSE134384", destdir = resfolder, GSEMatrix =TRUE)
show(gset_1)
show(pData(phenoData(gset_1)))


## ANALYSIS
## https://www.bioconductor.org/packages/devel/workflows/vignettes/rnaseqGene/inst/doc/rnaseqGene.html
## https://www.bioconductor.org/packages/devel/workflows/vignettes/RNAseq123/inst/doc/limmaWorkflow.html

## Fetch Data
url      = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63310&format=file"
destfile = "/mnt/space/Data/2.seq-data/GSE63310_RAW.tar"
utils::download.file(url, destfile=destfile, mode="wb") 
utils::untar(destfile, exdir = ".")

## After untarring and unzipping downsloaded files
setwd("/mnt/space/Data/2.seq-data/")
files = c("GSM1545535_10_6_5_11.txt", "GSM1545536_9_6_5_11.txt", "GSM1545538_purep53.txt", "GSM1545539_JMS8-2.txt", 
            "GSM1545540_JMS8-3.txt", "GSM1545541_JMS8-4.txt", "GSM1545542_JMS8-5.txt", "GSM1545544_JMS9-P7c.txt", 
            "GSM1545545_JMS9-P8c.txt")
x <- readDGE(files, columns=c(1,3))
class(x)
dim(x)


## Organize Sample Names
samplenames  = substring(colnames(x), 12, nchar(colnames(x)))
samplenames

colnames(x)  = samplenames
group = as.factor(c("LP", "ML", "Basal", "Basal", "ML", "LP", 
                     "Basal", "ML", "LP"))
x$samples$group = group
lane = as.factor(rep(c("L004","L006","L008"), c(3,4,2)))
x$samples$lane = lane
x$samples


## Organize Gene Annotations
geneid = rownames(x)
genes  = select(Mus.musculus, keys=geneid, columns=c("SYMBOL", "TXCHROM"), keytype="ENTREZID")
head(genes)


## Resolve Anno for Duplicated genes
genes <- genes[!duplicated(genes$ENTREZID),]
x$genes <- genes
x


## NORMALIZATION
cpm  =  cpm(x)
lcpm = cpm(x, log=TRUE)
x$samples

L =  mean(x$samples$lib.size) * 1e-6
M =  median(x$samples$lib.size) * 1e-6
c(L, M)
summary(lcpm)


## Remove lowly exoressed
table(rowSums(x$counts==0)==9)
keep.exprs <- filterByExpr(x, group=group)
x <- x[keep.exprs,, keep.lib.sizes=FALSE]
dim(x)


## PLOT
lcpm.cutoff <- log2(10/M + 2/L)
library(RColorBrewer)
nsamples <- ncol(x)
col <- brewer.pal(nsamples, "Paired")
par(mfrow=c(1,2))
plot(density(lcpm[,1]), col=col[1], lwd=2, ylim=c(0,0.26), las=2, main="", xlab="")
title(main="A. Raw data", xlab="Log-cpm")
abline(v=lcpm.cutoff, lty=3)
for (i in 2:nsamples){
den <- density(lcpm[,i])
lines(den$x, den$y, col=col[i], lwd=2)
}
legend("topright", samplenames, text.col=col, bty="n")
lcpm <- cpm(x, log=TRUE)
plot(density(lcpm[,1]), col=col[1], lwd=2, ylim=c(0,0.26), las=2, main="", xlab="")
title(main="B. Filtered data", xlab="Log-cpm")
abline(v=lcpm.cutoff, lty=3)
for (i in 2:nsamples){
den <- density(lcpm[,i])
lines(den$x, den$y, col=col[i], lwd=2)
}
legend("topright", samplenames, text.col=col, bty="n")


## Normalizing Gene Distributions
x <- calcNormFactors(x, method = "TMM")
x$samples
x[1:10,]

lcpm <- cpm(x, log=TRUE)
colnames(lcpm)
lcpm[1:10,]