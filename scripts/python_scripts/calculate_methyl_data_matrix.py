# use hmark_env2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNetCV, ElasticNet


source_path = os.path.abspath(os.path.join('../..'))
if source_path not in sys.path:
    sys.path.append(os.path.join(source_path, 'source'))
    
# read source files–––
from histone_mark_dataset import histoneMarkDataset
from methylation_dataset import methylationDataset

# read in two commandline arguments
start_sample_number = int(sys.argv[1])
end_sample_number = int(sys.argv[2])
# count available cores
n_cores = os.cpu_count()
n_cores = 10

# CEEHRC
"""metadata_df = pd.read_csv("/cellar/users/zkoch/histone_mark_proj/data/CEEHRC/metadata_final.csv", index_col=0)
ceehrc_methyl = methylationDataset(
    name = "ceehrc_methyl",
    species = "human",
    bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/CEEHRC/methyl_bed",
    regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
    gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz",
    metadata_df = metadata_df,
    n_cores = n_cores,
    use_cached_data_matrix = False,
    overwrite_cached_data_matrix = True,
    sample_name_part = 0,
    split_on = '.',
    start_sample_number = start_sample_number,
    end_sample_number = end_sample_number,
    )
"""
# mouse encode
"""mouse_encode_methyl = methylationDataset(
    name = "mouse_encode_methyl",
    species = "mus_musculus",
    bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/mouse_bed_files/methyl",
    regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
    gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
    metadata_fn = os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"mouse_methyl_metadata_finalFOREAL.parquet"),
    n_cores = n_cores,
    use_cached_data_matrix = False,
    overwrite_cached_data_matrix = True,
    start_sample_number = start_sample_number,
    end_sample_number = end_sample_number
)
"""

"""metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"human_methyl_metadata_fixed.parquet")).set_index('File accession')
methyl_human_encode = methylationDataset(
    name = "methyl_human_encode",
    species = "homo_sapiens",
    bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/human_bed_files/methyl",
    regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
    gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz", 
    metadata_df = metadata_df,
    n_cores = n_cores,
    use_cached_data_matrix = False,
    overwrite_cached_data_matrix = True,
    start_sample_number = start_sample_number,
    end_sample_number = end_sample_number
)"""

# human blueprint
def map_age_range_to_midpoint(age_range):
    if age_range == '-' or pd.isna(age_range):
        return np.nan
    elif age_range == '90+':
        return 92.5 # assume a little higher than 90, since most probably are
    elif ' - ' in str(age_range):
        start, end = map(float, age_range.split(' - '))
        return (start + end) / 2
    else:
        return float(age_range)  # For single numbers
    
metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/blueprint/metadata', f"blueprint_metadata_final.parquet")).set_index('EXPERIMENT_ID')
metadata_df.rename(columns = {'DONOR_AGE': 'age_years', 'SAMPLE_NAME': 'donor', 'TISSUE_TYPE': 'tissue'}, inplace = True)
# Convert age ranges to midpoints
metadata_df['age_years'] = metadata_df['age_years'].apply(map_age_range_to_midpoint)
# dropna
metadata_df = metadata_df.dropna(subset = ['age_years'])
# rename 
human_blueprint_methyl = methylationDataset(
    name = "human_blueprint_methyl",
    species = "homo_sapiens",
    bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/blueprint/methyl",
    regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
    gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz", 
    metadata_df = metadata_df,
    n_cores = n_cores,
    use_cached_data_matrix = False,
    overwrite_cached_data_matrix = True,
    start_sample_number = start_sample_number,
    end_sample_number = end_sample_number
    )