# Histone Mark Epigenetic Aging Project

## Abstract

Epigenetic remodeling is a hallmark of aging, yet which epigenetic layers are most affected during aging – and the extent to which they are interrelated – is not well understood. Here, we perform a comprehensive analysis of epigenetic aging encompassing 6 histone marks and DNA methylation measured across 12 tissues from >1,000 humans and mice. We identify a synchronized pattern of age-related changes across these epigenetic layers, with all changes converging upon a common set of genes. Notably, an epigenetic clock based on these genes can accurately predict age using data from any layer (Spearman ρ: 0.70 in humans, 0.81 in mice). Applying this "pan-epigenetic" clock, we observe that histone modification and DNA methylation profiles agree in prediction of which individuals are aging more rapidly or slowly. These results demonstrate that epigenetic modifications are subject to coordinated remodeling over the lifespan, offering a unified view of epigenetic aging.

## Project Overview

This project analyzes epigenetic aging patterns across multiple histone modifications (H3K27me3, H3K4me1, H3K4me3, H3K9me3, H3K27ac, H3K36me3) and DNA methylation in human and mouse tissues. The analysis includes data from multiple public datasets and aims to develop a unified "pan-epigenetic" clock that can predict biological age from any epigenetic layer.

## Environment Setup

### Python Environment

Create and activate the conda environment:

```bash
conda env create -f envs/histone_env.yaml
conda activate histone_env
```

### R Dependencies

Install required R packages:

```bash
Rscript install_packages.R
```

## Directory Structure

```
├── data/                    # Raw and processed datasets
│   ├── encode/             # ENCODE project data (human/mouse ChIP-seq, WGBS)
│   ├── meer_2018/          # Meer et al. 2018 dataset
│   ├── petkovich_2017/     # Petkovich et al. 2017 dataset
│   ├── signal_2024/        # Signal et al. 2024 dataset
│   ├── yang_2023/          # Yang et al. 2023 dataset
│   ├── hillje_2022/        # Hillje et al. 2022 dataset
│   ├── stubbs_2017/        # Stubbs et al. 2017 dataset
│   ├── methbank4/          # MethBank 4.0 methylation data
│   ├── CEEHRC/             # Canadian Epigenetics, Environment and Health Research Consortium
│   ├── ROSMAP/             # Religious Orders Study and Memory and Aging Project
│   ├── blueprint/          # BLUEPRINT Consortium data
│   ├── hannum/             # Hannum methylation clock data
│   └── ensembl_reference/  # Reference genome annotations
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 092624_data_aggregation.ipynb    # Data collection and preprocessing
│   ├── 100424_age_prediction.ipynb      # Age prediction model development
│   └── 010325_model_free.ipynb          # Model-free analysis approaches
├── scripts/                # Processing and analysis scripts
│   ├── python_scripts/     # Python analysis scripts
│   ├── r_scripts/         # R statistical analysis scripts
│   ├── bash_scripts/      # Shell scripts for data processing
│   └── submission_scripts/ # Cluster job submission scripts
├── utils/                  # Utility files and reference data
│   ├── ensembl_to_geneSymbol.tsv       # Gene ID conversion table
│   ├── illumina_cpg_450k_locations.csv # Illumina 450K array probe locations
│   └── refGene.txt.gz                  # Reference gene annotations
├── figures/                # Generated plots and visualizations
├── model_results/          # Model outputs and performance metrics
├── source/                 # Source code and additional scripts
└── envs/                   # Environment configuration files
    └── histone_env.yaml    # Conda environment specification
```

## Data Sources

The project integrates epigenetic data from multiple sources:

- **ENCODE Project**: ChIP-seq data for 6 histone marks and WGBS data from human and mouse tissues
- **GEO Datasets**: Multiple published studies with age-related epigenetic profiles
- **Methylation Arrays**: Illumina 450K and EPIC array data from aging studies
- **Reference Data**: Ensembl gene annotations and genomic coordinates

## Key Analysis Steps

1. **Data Aggregation** (`092624_data_aggregation.ipynb`): Collection and standardization of multi-modal epigenetic datasets
2. **Age Prediction** (`100424_age_prediction.ipynb`): Development of pan-epigenetic age prediction models
3. **Model-Free Analysis** (`010325_model_free.ipynb`): Non-parametric approaches to identify aging signatures

## Key Features

- Integration of 6 histone modifications and DNA methylation data
- Cross-species analysis (human and mouse)
- Multi-tissue profiling across 12 different tissue types
- Development of unified epigenetic aging clocks
- Analysis of >1,000 individual samples

## File Types and Data Formats

The project handles various genomic data formats (see `.gitignore` for complete list):
- **Peak files**: BED format for ChIP-seq peaks
- **Signal tracks**: BigWig files for continuous signal data
- **Methylation data**: CSV/TSV files with CpG methylation values
- **Sequencing data**: FASTQ, BAM, CRAM files
- **Processed results**: Parquet, JSON, and compressed formats

## Usage

1. Set up the environment as described above
2. Download required datasets (see `data/README.md` for specific instructions)
3. Run data aggregation notebooks to preprocess the data
4. Execute age prediction analysis to generate epigenetic clocks
5. Use model-free approaches for additional biological insights

## Contributing

This project follows standard bioinformatics best practices:
- Use conda for Python package management
- Use Bioconductor for R package management
- Follow PEP 8 for Python code style
- Document all analysis steps in Jupyter notebooks

## Contact

For questions about this analysis or to request access to processed datasets, please refer to the associated publication. 