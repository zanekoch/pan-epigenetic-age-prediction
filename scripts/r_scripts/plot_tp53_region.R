# Install required packages if not present
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if (!requireNamespace("karyoploteR", quietly = TRUE))
    BiocManager::install("karyoploteR")

if (!requireNamespace("GenomicRanges", quietly = TRUE)) 
    BiocManager::install("GenomicRanges")

if (!requireNamespace("dplyr", quietly = TRUE))
    install.packages("dplyr")

if (!requireNamespace("TxDb.Hsapiens.UCSC.hg19.knownGene", quietly = TRUE))
    BiocManager::install("TxDb.Hsapiens.UCSC.hg19.knownGene")

if (!requireNamespace("org.Hs.eg.db", quietly = TRUE))
    BiocManager::install("org.Hs.eg.db")
    
if (!requireNamespace("BSgenome.Hsapiens.UCSC.hg19", quietly = TRUE))
    BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")

# Load required libraries
library(karyoploteR)
library(GenomicRanges)
library(dplyr)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(org.Hs.eg.db)
library(BSgenome.Hsapiens.UCSC.hg19)

# Create SVG device
svg("/cellar/users/zkoch/histone_mark_proj/figures/model_free/tp53_region_peaks.svg", 
    width = 12, height = 8) 

TP53.region <- toGRanges("chr17:7,550,000-7,620,000")
kp <- plotKaryotype(zoom = TP53.region)

genes.data <- makeGenesDataFromTxDb(TxDb.Hsapiens.UCSC.hg19.knownGene,
                                    karyoplot=kp,
                                    plot.transcripts = TRUE, 
                                    plot.transcripts.structure = TRUE)

genes.data <- addGeneNames(genes.data)
genes.data <- mergeTranscripts(genes.data)

base.url <- "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/"
histone.marks <- c(H3K4me3="wgEncodeBroadHistoneK562H3k4me3StdSig.bigWig",
                   H3K36me3="wgEncodeBroadHistoneK562H3k36me3StdSig.bigWig",
                   H3K27ac="wgEncodeBroadHistoneK562H3k27acStdSig.bigWig",
                   H3K9me3="wgEncodeBroadHistoneK562H3k9me3StdSig.bigWig",
                   H3K4me1="wgEncodeBroadHistoneK562H3k4me1StdSig.bigWig",
                   H3K27me3="wgEncodeBroadHistoneK562H3k27me3StdSig.bigWig",
                   DNAm="wgEncodeBroadHistoneH1hescH3k4me1StdSig.bigWig")

pp <- getDefaultPlotParams(plot.type=1)
pp$leftmargin <- 0.15
pp$topmargin <- 15
pp$bottommargin <- 15
pp$ideogramheight <- 5
pp$data1inmargin <- 10

kp <- plotKaryotype(zoom = TP53.region, cex=2, plot.params = pp)
kpAddBaseNumbers(kp, tick.dist = 10000, minor.tick.dist = 2000,
                 add.units = TRUE, cex=1.3, digits = 6)
kpPlotGenes(kp, data=genes.data, r0=0, r1=0.1, gene.name.cex = 2)

#Histone marks
total.tracks <- length(histone.marks)+length(DNA.binding)
out.at <- autotrack(1:length(histone.marks), total.tracks, margin = 0.3, r0=0.23)

for(i in seq_len(length(histone.marks))) {
  bigwig.file <- paste0(base.url, histone.marks[i])
  at <- autotrack(i, length(histone.marks), r0=out.at$r0, r1=out.at$r1, margin = 0.1)
  kp <- kpPlotBigWig(kp, data=bigwig.file, ymax="visible.region",
                     r0=at$r0, r1=at$r1, col = "cadetblue2")
  computed.ymax <- ceiling(kp$latest.plot$computed.values$ymax)
  kpAxis(kp, ymin=0, ymax=computed.ymax, numticks = 2, r0=at$r0, r1=at$r1)
  kpAddLabels(kp, labels = names(histone.marks)[i], r0=at$r0, r1=at$r1, 
              cex=1.6, label.margin = 0.035)
}

dev.off()