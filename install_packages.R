# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# Install karyoploteR
BiocManager::install("karyoploteR") 