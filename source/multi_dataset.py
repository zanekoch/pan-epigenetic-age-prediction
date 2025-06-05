import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import subprocess
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import warnings
from scipy.sparse import csr_matrix
from xgboost import XGBRegressor
from palettable.cartocolors.diverging import Fall_3, Geyser_3, TealRose_3
from collections import defaultdict


import palettable
# use Prism_10
from palettable.cartocolors.qualitative import Prism_10, Safe_8
dataset_colors = Safe_8.mpl_colors
omics_color_palette = Prism_10.mpl_colors
# create a mapping from datasets to colors
dataset_colors = {
    'encode-human': dataset_colors[0],
    'blueprint-human': dataset_colors[1],
    'encode-mouse': dataset_colors[2],
    'signal-mouse': dataset_colors[3],
    'yang-mouse': dataset_colors[4],
    'stubbs-mouse': dataset_colors[5]
    
}
# create a mapping from hmarks to colors
omics_colors = {
    'H3K4me3': omics_color_palette[0],
    'H3K27ac': omics_color_palette[1],
    'H3K27me3': omics_color_palette[2],
    'H3K4me1': omics_color_palette[3],
    'H3K36me3': omics_color_palette[4],
    'H3K9me3': omics_color_palette[5],
    'H3K9ac': omics_color_palette[6],
    'H4K20me1': omics_color_palette[7],
    'H3K9/14ac': omics_color_palette[8],
    'H2A.Zac': 'black',
    'methylation': omics_color_palette[9]
}

# Set global plotting parameters
plt.rcParams.update({
    # Figure aesthetics
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    # Font sizes
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 8,
    
    # use arial
    'font.family': 'sans-serif',
    
    # Ensure text is editable in illustrator
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    
    # Grid settings
    'axes.grid': False,
    
    # Legend settings
    'legend.frameon': False,
    
    'axes.spines.top': False, # Show top spine
    'axes.spines.right': False, # Show right spine
})

TISSUE_MAP = {
    # Smooth muscle of the digestive tract
    "gastroesophageal sphincter": "muscle",
    
    # Pancreas
    "pancreas": "pancreas",
    "body of pancreas": "pancreas",
    "endocrine pancreas": "pancreas",
    
    # Heart tissue
    "heart right ventricle": "heart",
    "heart": "heart",
    "heart left ventricle": "heart",
    "right cardiac atrium": "heart",
    "left ventricle myocardium inferior": "heart",
    "right atrium auricular region": "heart",
    "Heart": "heart",
    
    # Skin/epithelial
    "lower leg skin": "skin",
    "suprapubic skin": "skin",
    "skin epidermis": "skin",
    "Basal cells": "skin",      # Assumed skin basal epithelial cells
    "Basal cell": "skin",       # Same assumption as above
    
    # Nervous system (brain/spinal/neuronal)
    "tibial nerve": "brain",  # Peripheral nerve, grouped under broader 'nervous system'
    "neuronal stem cell": "brain",
    "spinal cord": "brain",
    "Cortex": "brain",
    "cingulate gyrus": "brain",
    "temporal lobe": "brain",
    "dorsolateral prefrontal cortex": "brain",
    "caudate nucleus": "brain",
    "germinal matrix": "brain",
    "neuroepithelial stem cell": "brain",
    "neural progenitor cell": "brain",
    "substantia nigra": "brain",
    "neuron": "brain",
    "mid-neurogenesis radial glial cells": "brain",
    "layer of hippocampus": "brain",
    "brain": "brain",
    "angular gyrus": "brain",
    "cerebellum": "brain",
    "Brain-Frontal Lobe-Right": "brain",
    "Brain-Frontal Lobe-Left": "brain",
    "Brain-right temporal": "brain",
    "Brain-Temporal lobe-Left": "brain",
    "Brain-Occipetal Lobe-Right": "brain",
    "neurosphere": "brain",     # culture of neural cells
    "radial glial cell": "brain",
    
    # Digestive tract (esophagus, stomach, small/large intestine)
    "esophagus": "digestive",
    "esophagus squamous epithelium": "digestive",
    "esophagus muscularis mucosa": "digestive",
    "small intestine": "digestive",
    "duodenal mucosa": "digestive",
    "large intestine": "digestive",
    "stomach": "stomach",
    "mucosa of stomach": "stomach",
    
    # Colon (including large-intestine subsections)
    "transverse colon": "colon",
    "sigmoid colon": "colon",
    "colonic mucosa": "colon",
    "mucosa of descending colon": "colon",
    "mucosa of rectum": "colon",
    "Large Intestine-Colon-Ascending (Right)": "colon",
    "Large Intestine-Colon-Rectosigmoid": "colon",
    "Large Intestine-Colon": "colon",
    
    # Immune/Blood/Lymphatic
    "thymus": "immune",
    "spleen": "immune",
    "tonsil": "immune",
    "CD19+ cells (B lymphocytes)": "immune",
    "B cells": "immune",
    "Peyer's patch": "immune",
    "CD4 Naive": "immune",
    
    # Blood
    "cord blood": "blood",
    "venous blood": "blood",
    "bone marrow": "blood",
    "Whole Blood": "blood",
    
    # Muscle tissues
    "psoas muscle": "muscle",
    "muscle of leg": "muscle",
    "muscle of trunk": "muscle",
    "smooth muscle cell": "muscle",
    "stomach smooth muscle": "muscle",
    "skeletal muscle tissue": "muscle",
    "gastrocnemius medialis": "muscle",
    "muscle layer of duodenum": "muscle",
    "muscle layer of colon": "muscle",
    "rectal smooth muscle tissue": "muscle",
    
    # Lung
    "lung": "lung",
    "Lung": "lung",
    "upper lobe of left lung": "lung",
    "upper lobe of right lung": "lung",
    "lower lobe of left lung": "lung",
    "lower lobe of right lung": "lung",
    "left lung": "lung",
    
    # Liver
    "liver": "liver",
    "right lobe of liver": "liver",
    "Liver": "liver",
    "hepatocyte": "liver",
    
    # Breast
    "breast epithelium": "breast",
    "Luminal progenitor cell": "breast",
    "Luminal progenitor": "breast",
    "Luminal cells": "breast",
    "Luminal cell": "breast",
    
    # Reproductive organs
    "ovary": "reproductive",
    "testis": "reproductive",
    "uterus": "reproductive",
    "vagina": "reproductive",
    
    # Adrenal gland
    "adrenal gland": "adrenal gland",
    
    # Thyroid
    "thyroid gland": "thyroid",
    "Thyroid": "thyroid",
    
    # Adipose
    "adipose tissue": "adipose",
    "subcutaneous abdominal adipose tissue": "adipose",
    "adipocyte": "adipose",
    
    # Vascular (arteries, aorta, coronary)
    "aorta": "vascular",
    "ascending aorta": "vascular",
    "thoracic aorta": "vascular",
    "tibial artery": "vascular",
    "coronary artery": "vascular",
    
    # Kidney
    "kidney": "kidney",
    
    # Bladder
    "urinary bladder": "bladder",
    
    # Placenta / related
    "placenta": "placenta",
    "chorionic villus": "placenta",
    "chorion": "placenta",
    "placental basal plate": "placenta",
    "amnion": "placenta",
    "trophoblast": "placenta",
    
    # Prostate
    "prostate gland": "prostate",
    
    # Parathyroid or other endocrine
    "parathyroid adenoma": "endocrine",
    
    # Cell lines / cultured lines
    "HeLa-S3": "cell_line",
    "H9": "cell_line",
    "SK-N-SH": "cell_line",
    "HepG2": "cell_line",
    "A549": "cell_line",
    "K562": "cell_line",
    "IMR-90": "cell_line",
    "GM23248": "cell_line",
    "Stromal cell": "cell_line",
    "Stromal cells": "cell_line",
    "mesenchymal stem cell": "cell_line",
    "MG63": "cell_line",
    "NB4": "cell_line",
    "RWPE2": "cell_line",
    "SJCRH30": "cell_line",
    "NT2/D1": "cell_line",
    "OCI-LY3": "cell_line",
    "PC-3": "cell_line",
    "HAP-1": "cell_line",
    "OCI-LY1": "cell_line",
    "Loucy": "cell_line",
    "AG10803": "cell_line",
    "LNCaP clone FGC": "cell_line",
    "SJSA1": "cell_line",
    "Karpas-422": "cell_line",
    "MCF-7": "cell_line",
    "Panc1": "cell_line",
    "SU-DHL-6": "cell_line",
    "iPS-11a": "cell_line",
    "OCI-LY7": "cell_line",
    "WERI-Rb-1": "cell_line",
    "GM23338": "cell_line",
    "HCT116": "cell_line",
    "DOHH2": "cell_line",
    "A673": "cell_line",
    "iPS-20b": "cell_line",
    "ACC112": "cell_line",
    "NCI-H929": "cell_line",
    "AG04450": "cell_line",
    "KMS-11": "cell_line",
    "iPS-15b": "cell_line",
    "DND-41": "cell_line",
    "AG04449": "cell_line",
    "MM.1S": "cell_line",
    "C4-2B": "cell_line",
    "RWPE1": "cell_line",
    "BE2C": "cell_line",
    "SK-N-MC": "cell_line",
    "GM08714": "cell_line",
    "Jurkat, Clone E6-1": "cell_line",
    "Caco-2": "cell_line",
    "iPS-18c": "cell_line",
    "AG09309": "cell_line",
    "HL-60": "cell_line",
    "VCaP": "cell_line",
    
    "muscle layer of colon": "muscle",
    "mucosa of rectum": "colon",
    "iPS-18a": "cell_line",
    "mid-neurogenesis radial glial cells": "brain",
    "mucosa of descending colon": "colon",
    "Lung fibroblasts": "special",          # cultured fibroblasts from lung
    "Kidney fibroblasts": "special",        # cultured fibroblasts from kidney
    "iPSCs from kidney fibroblasts": "special",
    "iPSCs from lung fibroblasts": "special",
    "colonic mucosa": "colon",
    "esophagus muscularis mucosa": "digestive",
    "Whole Blood": "blood",
}


class MultiDataset:
    def __init__(
        self, 
        datasets: list,
        nan_policy: str = 'fill'
        ):
        """
        Initialize MultiDataset with a list of dataset objects
        
        Args:
            datasets (list[HistoneMarkDataset, MethylationDataset]): List of dataset objects (histone mark and/or methylation datasets)
        """
        self.datasets = datasets
        self.nan_policy = nan_policy
        self.exclude_from_training = []
        # first, harmonize the feature names to all be with respect to human genes
        self._harmonize_feature_names()        
        self._combine_datasets()
        self._scale_age_relative_to_lifespan()
        self.trained_models = {}
        self.feature_names = {}
        
    def _harmonize_feature_names(self):
        """
        Harmonizes the feature names to all be with respect to human genes
        """
        # read in mapper
        mouse_to_human_gene_map = pd.read_csv(
            '../data/ensembl_reference/human_mouse_ensembl_genes.txt.gz', sep='\t'
            )
        mouse_to_human_gene_map.dropna(subset = 'Mouse gene stable ID', inplace = True)
        mouse_to_human_gene_map.set_index('Mouse gene stable ID', inplace = True)
        mapper = mouse_to_human_gene_map['Gene stable ID'].to_dict()
        
        for dset in self.datasets:
            # convert and subset 
            if dset.species == 'mouse' or dset.species == 'mus_musculus':
                # drop meta_cols
                meta_cols = dset.data_matrix_w_meta[dset.meta_cols]
                dset.data_matrix_w_meta.drop(columns = dset.meta_cols, inplace = True)
                try:
                    # split each column on :
                    dset.data_matrix_w_meta.columns = [col.split(':')[1] for col in dset.data_matrix_w_meta.columns]
                    # map the columns to human genes
                    dset.data_matrix_w_meta.columns = [mapper[col] if col in mapper.keys() else np.nan for col in dset.data_matrix_w_meta.columns]
                except:
                    # skipping mapping bc already done
                    pass 
                # remove columns that are nan
                dset.data_matrix_w_meta = dset.data_matrix_w_meta.loc[:, ~dset.data_matrix_w_meta.columns.isna()]
                # resolve duplicated columns by averaging
                dset.data_matrix_w_meta = dset.data_matrix_w_meta.T.groupby(dset.data_matrix_w_meta.columns).mean().T
                # add the meta_cols back
                dset.data_matrix_w_meta = pd.concat([dset.data_matrix_w_meta, meta_cols], axis = 1)
                # if doesn't have species column, add it
                dset.data_matrix_w_meta['species'] = 'mouse'
                if 'species' not in dset.meta_cols:
                    dset.meta_cols.append('species')
                # if doesn't have dataset column, add it
                dset.data_matrix_w_meta['dataset'] = dset.name
                if 'dataset' not in dset.meta_cols:
                    dset.meta_cols.append('dataset')
            elif dset.species == 'human' or dset.species == 'homo_sapiens':
                # remove meta_cols
                meta_cols = dset.data_matrix_w_meta[dset.meta_cols]
                dset.data_matrix_w_meta.drop(columns = dset.meta_cols, inplace = True)
                try:
                    # split each column on :
                    dset.data_matrix_w_meta.columns = [col.split(':')[1] for col in dset.data_matrix_w_meta.columns]
                    # remove columns not starting with ENSG
                    dset.data_matrix_w_meta = dset.data_matrix_w_meta.loc[:, dset.data_matrix_w_meta.columns.str.startswith('ENSG')]
                except:
                    # skipping mapping bc already done
                    pass 
                # add the meta_cols back
                dset.data_matrix_w_meta = pd.concat([dset.data_matrix_w_meta, meta_cols], axis = 1)
                # if doesn't have species column, add it
                dset.data_matrix_w_meta['species'] = 'human'
                if 'species' not in dset.meta_cols:
                    dset.meta_cols.append('species')
                # if doesn't have dataset column, add it
                dset.data_matrix_w_meta['dataset'] = dset.name
                if 'dataset' not in dset.meta_cols:
                    dset.meta_cols.append('dataset')
            else:
                raise ValueError(f"Species {dset.species} not supported")
        
    def _combine_datasets(self):
        """
        Combines data_matrix_w_meta, metadata_dfs, and meta_cols from all datasets.
        If both methylation and histone datasets are present, combines them separately
        and scales before final combination.
        
        Returns:
            dict: Combined dataset information
        """
        # Separate datasets by type
        methyl_datasets = [d for d in self.datasets if hasattr(d, 'bedmethyl_dir')]
        histone_datasets = [d for d in self.datasets if not hasattr(d, 'bedmethyl_dir')]
        
        if len(methyl_datasets) > 0 and len(histone_datasets) > 0:
            # First combine methylation datasets, keeping only the intersection of features
            methyl_combined = pd.concat(
                [d.data_matrix_w_meta for d in methyl_datasets], 
                axis=0, join='inner'
            )
            methyl_combined = methyl_combined.copy()
            # drop rows with duplicate index
            #methyl_combined = methyl_combined[~methyl_combined.index.duplicated(keep = 'first')]
            
            # Combine histone datasets, keeping only the intersection of features
            histone_combined = pd.concat(
                [d.data_matrix_w_meta for d in histone_datasets],
                axis=0, join='inner'
            )
            histone_combined = histone_combined.copy()
            # drop rows with duplicate index
            #histone_combined = histone_combined[~histone_combined.index.duplicated(keep = 'first')]
            
            # go through each dataset's meta_cols, if any are not in methyl or histone_combined, add those columns in methyl_combined and histone_combined
            for dset in methyl_datasets:
                for col in dset.meta_cols:
                    if col not in methyl_combined.columns:
                        methyl_combined = pd.concat([methyl_combined, dset.data_matrix_w_meta[[col]]], axis = 1)
            for dset in histone_datasets:
                for col in dset.meta_cols:
                    if col not in histone_combined.columns:
                        histone_combined = pd.concat([histone_combined, dset.data_matrix_w_meta[[col]]], axis = 1)
                        
            # Get meta columns to exclude from scaling
            meta_cols = []
            for dset in self.datasets:
                this_dset_meta_cols = set(dset.meta_cols)
                meta_cols.extend(this_dset_meta_cols)
            self.meta_cols = list(set(meta_cols))
            
            # Scale methylation and histone data separately
            methyl_features = [col for col in methyl_combined.columns if col not in self.meta_cols]
            histone_features = [col for col in histone_combined.columns if col not in self.meta_cols]
            methyl_combined[methyl_features] = methyl_combined[methyl_features] / methyl_combined[methyl_features].max()
            histone_combined[histone_features] = histone_combined[histone_features] / histone_combined[histone_features].max()
            
            # add histone_mark column to methylation 
            methyl_combined['histone_mark'] = 'methylation'
            
            # Combine scaled datasets
            self.combined_data_matrix = pd.concat([methyl_combined, histone_combined], axis=0, join='outer')
            
            # make sure meta_cols are in the combined data matrix
            self.meta_cols = self.combined_data_matrix.columns.intersection(self.meta_cols).tolist()
        else:
            # If only one type, combine normally
            self.combined_data_matrix = pd.concat(
                [dset.data_matrix_w_meta for dset in self.datasets], 
                axis=0, join='inner'
            )
            
            # Get meta columns
            meta_cols = []
            for dset in self.datasets:
                this_dset_meta_cols = set(dset.meta_cols)
                shared_with_combined = this_dset_meta_cols.intersection(set(self.combined_data_matrix.columns))
                meta_cols.extend(shared_with_combined)
            self.meta_cols = list(set(meta_cols))
            
        # drop rows where age_years is 0
        self.combined_data_matrix = self.combined_data_matrix[self.combined_data_matrix['age_years'] != 0]
        # drop rows that are all 0, excluding meta_cols
        self.combined_data_matrix = self.combined_data_matrix[
            ~((self.combined_data_matrix.drop(columns = self.meta_cols) == 0).all(axis = 1))
            ]
        # drop columns that are all 0, excluding meta_cols
        feature_cols = [col for col in self.combined_data_matrix.columns if col not in self.meta_cols]
        cols_to_keep = [col for col in feature_cols if self.combined_data_matrix[col].sum() != 0]
        self.combined_data_matrix = self.combined_data_matrix[cols_to_keep + self.meta_cols]
        # Copy to defragment
        self.combined_data_matrix = self.combined_data_matrix.copy()
        # make sure meta_cols are in the combined data matrix
        self.meta_cols = self.combined_data_matrix.columns.intersection(self.meta_cols).tolist()
        # fill nan with column mean, excluding meta_cols
        feature_cols = [col for col in self.combined_data_matrix.columns if col not in self.meta_cols]
        if self.nan_policy == 'fill':
            self.combined_data_matrix.loc[:, feature_cols] = self.combined_data_matrix.loc[:, feature_cols].fillna(
                self.combined_data_matrix.loc[:, feature_cols].mean()
                )
        elif self.nan_policy == 'drop':
            # First get the columns to keep (those without any NaN in feature_cols)
            cols_to_keep = feature_cols.copy()
            cols_to_keep = [col for col in cols_to_keep if not self.combined_data_matrix[col].isna().any()]
            # Add back the meta columns
            cols_to_keep.extend(self.meta_cols)
            # Update the DataFrame with only the kept columns
            self.combined_data_matrix = self.combined_data_matrix[cols_to_keep]        
        else:
            raise ValueError(f"Invalid nan policy: {self.nan_policy}")
        
        # create general_tissue column
        self.combined_data_matrix['general_tissue'] = self.combined_data_matrix['tissue'].map(TISSUE_MAP)
        self.meta_cols.append('general_tissue')

        
    def _scale_age_relative_to_lifespan(self):
        """
        Scales age relative to lifespan
        """
        max_lifespans = {
            'human': 120,
            'mouse': 4
        }
        # for each row, divide by the max lifespan for that species
        self.combined_data_matrix['age_scaled'] = self.combined_data_matrix.apply(
            lambda row: row['age_years'] / max_lifespans[row['species']], axis = 1
            )
        # -log(-log(x))
        self.combined_data_matrix['age_scaled_loglog'] = -np.log(-np.log(self.combined_data_matrix['age_scaled']))
        # copy to defragment
        self.combined_data_matrix = self.combined_data_matrix.copy()
        self.meta_cols.append('age_scaled')
        self.meta_cols.append('age_scaled_loglog')

    def minmax_scale_within_group(self, group_cols = ['dataset', 'histone_mark']):
        """
        MinMax scale features within groups
        
        Args:
            df: pandas DataFrame
            group_col: column name to group by
            exclude_cols: list of columns to exclude from scaling
        """
        from sklearn.preprocessing import MinMaxScaler
        
        cols_to_scale = [col for col in self.combined_data_matrix.columns if col not in self.meta_cols]

        grouper = self.combined_data_matrix.groupby(group_cols)
        
        scaled = []
        for name, group in grouper:
            # divide each column by the max value in that column
            group.loc[:, cols_to_scale] = group.loc[:, cols_to_scale] / group.loc[:, cols_to_scale].max()
            scaled.append(group.loc[:, cols_to_scale])
            
        # update the combined data matrix
        scaled_df = pd.concat(scaled)
        self.combined_data_matrix.loc[scaled_df.index, cols_to_scale] = scaled_df.loc[:, cols_to_scale]
        
        
    def predict(
        self, 
        target_column: str, 
        model, 
        result_column: str, 
        n_folds: int = 5,
        features: list = None,
        seperate_by_hmark: bool = True,
        n_pcs: int = -1,
        use_scrambled_for_training: bool = False
        ):
        """
        Perform N-fold cross-validation to predict the specified column using the specified model.
        Store the results in the specified column of self.data_matrix_w_meta and save the models to a dictionary.

        Parameters:
        -----------
        target_column : str
            The column to predict.
        model : sklearn-like model
            The model to use for prediction.
        result_column : str
            The column to store the prediction results.
        n_folds : int
            The number of folds for cross-validation.
        features : list, optional
            The features to use for prediction. If not specified, all features will be used.
        seperate_by_hmark : bool, optional
            If True, the data will be split by histone mark and a separate model will be trained for each.
        n_pcs: int, optional
            Number of principal components to use. If > 0, will compute PCs and use them as features.
        use_scrambled_for_training: bool, optional
            Whether to use scrambled data for training. If True, will use scrambled_data_matrix
            for training but still test on original data. Defaults to False.
        """
        # If using nonlinear PCs, compute them first
        if n_pcs > 0:
            print("Computing nonlinear PCs...", flush=True)
            self.compute_nonlinear_pca(
                n_components=n_pcs,
            )
            # Use the PC columns as the features for prediction
            features = [f'KPCA{i+1}' for i in range(n_pcs)]
            print(f"Using {n_pcs} nonlinear PCs as features", flush=True)
        
        # Prepare data
        y = self.combined_data_matrix[target_column]
        X = self.combined_data_matrix
        
        # drop columns (except for meta_cols) with any nan and count how many columns are left
        n_cols_before = X.shape[1]
        X_meta = X[self.meta_cols]
        X_non_meta = X.drop(columns = self.meta_cols)
        X_non_meta = X_non_meta.dropna(axis = 1)
        X = pd.concat([X_meta, X_non_meta], axis = 1)
        n_cols_after = X.shape[1]
        print(f"Dropped {n_cols_before - n_cols_after} columns with nans")
        
        # drop rows with nan in target_column
        n_rows_before = X.shape[0]
        X = X.dropna(subset = [target_column])
        n_rows_after = X.shape[0]
        print(f"Dropped {n_rows_before - n_rows_after} rows with nans in  target col: {target_column}")
        
        # subset the data matrix to the specified features
        if features is not None:
            X = X[features + self.meta_cols]
        
        # split the data matrix into a dictionary of data matrices, keyed by histone mark
        data_dict = {}
        if seperate_by_hmark:
            for hmark in X['histone_mark'].unique():
                subset = X.loc[X['histone_mark'] == hmark]
                # get the indices for samples to exclude from training
                excluded_indices = subset.index.isin(self.exclude_from_training)
                # Only include groups with sufficient non-excluded samples
                n_training_samples = len(subset[~excluded_indices])
                if n_training_samples >= n_folds:  # ensure at least n_folds samples
                    data_dict[hmark] = subset
                else:
                    print(f"Skipping histone mark {hmark} - insufficient training samples ({n_training_samples} < {n_folds})")
        else:
            data_dict = {'all': X}

        # drop the meta columns, except for 'donor'
        # so that we can keep samples from the same donors in the same fold
        meta_cols_to_drop = [col for col in self.meta_cols if col != 'donor']

        # train the model for each histone mark
        all_partition_predictions = {}
        for partition, data_df in data_dict.items():
            print(f"Training model for {partition}", flush=True)
            partition_predictions = self._do_kfold_cv(
                model = model, 
                X = data_df.drop(
                    columns=meta_cols_to_drop,
                    errors='ignore'
                    ), 
                y = data_df[target_column], 
                n_folds = n_folds, 
                result_column = result_column + '_' + partition,
                use_scrambled_for_training = use_scrambled_for_training
                )
            all_partition_predictions.update(partition_predictions)
        # update the data matrix with the predictions
        self.combined_data_matrix[result_column] = self.combined_data_matrix.index.map(all_partition_predictions)

        # Calculate and print the mean squared error
        try:
            mse = mean_squared_error(y, self.combined_data_matrix[result_column])
            r, p = pearsonr(y, self.combined_data_matrix[result_column])
            rho, p_s = spearmanr(y, self.combined_data_matrix[result_column])
            print(f"Mean Squared Error for {target_column}: {mse}")
            print(f"Pearson's r for {target_column}: {r}, p = {p}")
            print(f"Spearman's rho for {target_column}: {rho}, p = {p_s}")
        except:
            print(f"Could not calculate metrics for {target_column}")
        
        
    def _do_kfold_cv(self, model, X, y, n_folds, result_column, use_scrambled_for_training=False) -> dict:
        """
        Perform K-fold cross-validation.

        Parameters:
        -----------
        model : sklearn-like model
            The model to use for prediction.
        X : pandas.DataFrame
            The features to use for prediction.
        y : pandas.Series
            The target values to predict.
        kf : sklearn.model_selection.KFold
            The KFold object to use for cross-validation.
        result_column : str
            The column to store the prediction results.
        use_scrambled_for_training : bool
            Whether to use scrambled data for training. If True, will use scrambled_data_matrix
            for training but still test on original data. Defaults to False.
        
        Returns:
        --------
        y_pred_dict : dict
            A dictionary with the index of the test set as the key and the predicted value as the value.
        """
        self.trained_models[result_column] = []
        
        # Get indices of samples to exclude from training
        excluded_indices = X.index.isin(self.exclude_from_training)
        training_indices = ~excluded_indices
        
        if n_folds > 1:
            groups = X['donor']
            # check if we can split or don't have enough samples
            try:
                # Initialize StratifiedGroupKFold, balancing by age and grouping by donor
                sgf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
                classes = pd.qcut(y, 5).cat.codes
                splitter = sgf.split(X, classes, groups=groups)
                for _, (train_index, test_index) in enumerate(splitter):
                    break
                # reset the splitter
                splitter = sgf.split(X, classes, groups=groups)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Could not split data for {result_column}. Splitting by groups only instead")
                gf = GroupKFold(n_splits=n_folds)
                # instead just split by groups
                splitter = gf.split(X, groups=groups)
            
            # Get training samples and their groups for splitting
            # This extra step is necessary to exlude self.exclude_from_training samples
            X_train = X[training_indices]
            y_train = y[training_indices]
            groups = X_train['donor']
            
            # Perform cross-validation on training data
            y_pred_dict = {}
            classes = pd.qcut(y_train, 5).cat.codes
            #print("WARNING: Change 2 classes back to 5")
            sgf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splitter = sgf.split(X_train, classes, groups=groups)
            for _, (train_index, test_index) in enumerate(splitter):
                X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
                y_fold_train, _ = y_train.iloc[train_index], y_train.iloc[test_index]
                
                warnings.filterwarnings("ignore")
                # remove donor from the features
                X_fold_train = X_fold_train.drop(columns=['donor'])
                X_fold_test = X_fold_test.drop(columns=['donor'])
                
                # If using scrambled data for training, get the corresponding scrambled features
                if use_scrambled_for_training:
                    if not hasattr(self, 'scrambled_data_matrix'):
                        raise ValueError("Scrambled data matrix not found. Please run scramble_histone_mark first.")
                    # Get the scrambled features for training data only
                    X_fold_train = self.scrambled_data_matrix.loc[X_fold_train.index, X_fold_train.columns]
                
                # Convert to sparse matrix if using XGBoost
                if isinstance(model, XGBRegressor):
                    X_train_sparse = csr_matrix(X_fold_train.values)
                    X_test_sparse = csr_matrix(X_fold_test.values)
                    model = model.fit(X_train_sparse, y_fold_train)
                    y_pred = model.predict(X_test_sparse)
                else:
                    model = model.fit(X_fold_train, y_fold_train)
                    y_pred = model.predict(X_fold_test)
                
                # Store predictions for test fold
                y_pred_dict.update({X_fold_test.index[i]: y_pred[i] for i in range(len(y_pred))})
                
                # Also predict on excluded samples using this fold's model
                if any(excluded_indices):
                    X_excluded = X[excluded_indices].drop(columns=['donor'])
                    if isinstance(model, XGBRegressor):
                        X_excluded_sparse = csr_matrix(X_excluded.values)
                        y_pred_excluded = model.predict(X_excluded_sparse)
                    else:
                        y_pred_excluded = model.predict(X_excluded)
                    # Store predictions for excluded samples
                    for i, idx in enumerate(X[excluded_indices].index):
                        if idx not in y_pred_dict:
                            y_pred_dict[idx] = y_pred_excluded[i]
                        else:
                            # average with existing prediction, to end up with a predicted average across all folds
                            y_pred_dict[idx] = (y_pred_dict[idx] + y_pred_excluded[i]) / 2
                        
                
                # Save the trained model
                self.trained_models[result_column].append(model)
                self.feature_names[result_column] = X_fold_train.columns
        else:
            # For no cross-validation, train on all non-excluded samples
            X_train = X[training_indices].drop(columns=['donor'])
            y_train = y[training_indices]
            
            # If using scrambled data for training, get the corresponding scrambled features
            if use_scrambled_for_training and hasattr(self, 'scrambled_data_matrix'):
                # Get the scrambled features for training data only
                X_train = self.scrambled_data_matrix.loc[X_train.index, X_train.columns]
            
            if isinstance(model, XGBRegressor):
                X_train_sparse = csr_matrix(X_train.values)
                model = model.fit(X_train_sparse, y_train)
            else:
                model = model.fit(X_train, y_train)
            
            # Predict on all samples (both training and excluded) using original data
            X_all = X.drop(columns=['donor'])
            if isinstance(model, XGBRegressor):
                X_all_sparse = csr_matrix(X_all.values)
                y_pred = model.predict(X_all_sparse)
            else:
                y_pred = model.predict(X_all)
            
            y_pred_dict = {X.index[i]: y_pred[i] for i in range(len(y_pred))}
            self.trained_models[result_column].append(model)
            self.feature_names[result_column] = X_train.columns
            
        return y_pred_dict
    
    
    def predict_on_cluster(
        self,
        output_dir,
        result_column,
        target_column="age_years",
        features=None,
        n_folds=5,
        seperate_by_hmark = True,
        time="24:00:00",
        mem_gb=32,
        cpus=4,
        jobname = 'hstclck',
        model_type = 'ElasticNetCV',
        n_pcs = -1,
        use_scrambled_for_training = False
        ):
        """
        Submit prediction job to Slurm cluster as a class method
        
        Args:
            output_dir: Directory to save results and models
            result_column: Name for prediction column
            target_column: Column to predict
            features: List of features to use (defaults to all)
            n_folds: Number of folds for cross-validation
            seperate_by_hmark: If True, train a separate model for each histone mark
            time: Time limit for job (HH:MM:SS)
            mem_gb: Memory in GB
            cpus: Number of CPUs to request
            jobname: name of job
            model_type: Type of model to use (ElasticNetCV or XGBRegressor)
            n_pcs: Number of nonlinear PCs to use for prediction
            use_scrambled_for_training: Whether to use scrambled data for training. If True, will use scrambled_data_matrix
                for training but still test on original data. Defaults to False.
        """

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create python script for prediction
        script_content = f"""
import pickle
import os
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
import sys
source_path = os.path.abspath(os.path.join('/cellar/users/zkoch/histone_mark_proj'))
if source_path not in sys.path:
    sys.path.append(os.path.join(source_path, 'source'))
from histone_mark_dataset import histoneMarkDataset
from methylation_dataset import methylationDataset
from multi_dataset import MultiDataset

# Load data object
with open('{output_dir}/data_object.pkl', 'rb') as f:
    data_obj = pickle.load(f)

# Run prediction
results = data_obj.predict(
    target_column='{target_column}',
    model={model_type}(n_jobs={cpus}, random_state=42, verbose=1),
    result_column='{result_column}',
    n_folds={n_folds},
    features={features},
    seperate_by_hmark={seperate_by_hmark},
    n_pcs={n_pcs},
    use_scrambled_for_training={use_scrambled_for_training}
)

# Save predictions, which are in the combined_data_matrix dataframe
data_obj.combined_data_matrix['{result_column}'].to_csv('{output_dir}/predictions.csv')

# Save models
with open('{output_dir}/models.pkl', 'wb') as f:
    pickle.dump(data_obj.trained_models, f)

# Save feature names
with open('{output_dir}/feature_names.pkl', 'wb') as f:
    pickle.dump(data_obj.feature_names, f)
        """
        
        script_path = output_dir / "run_prediction.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
                    
        # Save data object
        with open(output_dir / "data_object.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        # Create slurm submission script
        slurm_script = f"""#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition=nrnb-compute
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
#SBATCH --job-name={jobname}

srun python {script_path}
        """
        
        slurm_path = output_dir / "submit.sh"
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        
        # Submit job
        subprocess.run(['sbatch', str(slurm_path)])
        print(f"Job submitted. Results will be saved to {output_dir}")
        
        # write the parameters of training to a file
        with open(output_dir / "training_parameters.txt", 'w') as f:
            f.write(f"target_column: {target_column}\n")
            f.write(f"result_column: {result_column}\n")
            f.write(f"n_folds: {n_folds}\n")
            f.write(f"seperate_by_hmark: {seperate_by_hmark}\n")
            f.write(f"use_scrambled_for_training: {use_scrambled_for_training}\n")
            # datasets
            f.write("Datasets:\n")
            for dset in self.datasets:
                f.write(f"{dset.name}\n")
            f.write(f"time: {time}\n")
            f.write(f"mem_gb: {mem_gb}\n")
            f.write(f"cpus: {cpus}\n")
            f.write(f"jobname: {jobname}\n")
        
    def read_cluster_predictions(
        self,
        out_dir: str,
        just_predictions: bool = False
        ):
        """
        Read in predictions from cluster jobs.

        Args:
            out_dir (str): Directory containing prediction results
            just_predictions (bool): If True, only read in predictions and not models or feature names
        """
        # Read predictions
        predictions = pd.read_csv(os.path.join(out_dir, 'predictions.csv'), index_col=0)
        # Read models
        if just_predictions == False:
            self.trained_models = pd.read_pickle(os.path.join(out_dir, 'models.pkl'))
            self.feature_names = pd.read_pickle(os.path.join(out_dir, 'feature_names.pkl'))
        # Update data matrix with predictions
        self.combined_data_matrix['pred_age'] = predictions
        
    def reverse_log_log_age_transform(self,pred_age_col: str):
        self.combined_data_matrix[pred_age_col] = np.exp(-np.exp(-self.combined_data_matrix[pred_age_col]))
        # multiply by 120 if human or 4 if mouse
        self.combined_data_matrix[pred_age_col] = self.combined_data_matrix.apply(
            lambda row: row[pred_age_col] * 120 if row['species'] == 'human' else row[pred_age_col] * 4, axis=1
            )
        
        
    def plot_predictions_scatter(
        self, 
        pred_col: str = '', 
        actual_age: str = 'age_scaled',
        hue_col: str = '',
        facet_by: str = 'histone_mark',
        subset_col: str = '',
        subset_val: str = ''
        ):
        """
        Create a faceted scatter plot of predicted vs actual age for each histone mark.
        
        Args:
            pred_col (str): Name of column containing predictions   
            actual_age (str): Name of column containing actual age values. Defaults to 'age_scaled'
            facet_by (str): Name of column to facet by. Defaults to 'histone_mark'
            subset_str: query string to subset the data matrix
        """

        if pred_col == '':
            # guess that the prediction column is the last column in the data matrix
            pred_col = self.combined_data_matrix.columns[-1]
            
      
        if facet_by == 'methylation':
            n_facets = 1
            n_cols = 1
            n_rows = 1
            facet_vals = ['methylation']
        else:
            # Get unique values for faceting
            facet_vals = self.combined_data_matrix[facet_by].unique()
            n_facets = len(facet_vals)
            # Calculate grid dimensions
            n_cols = min(4, n_facets)
            n_rows = int(np.ceil(n_facets / n_cols))
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.25*n_cols, 2*n_rows))
        if n_facets == 1:
            axes = [axes]  # Convert single axis to list for consistent handling
        elif n_rows > 1:
            axes = axes.flatten()
        
        # Plot each facet
        for i, (facet_val, ax) in enumerate(zip(facet_vals, axes)):
            if facet_by == 'histone_mark':
                subset = self.combined_data_matrix[
                    (self.combined_data_matrix[facet_by] == facet_val)
                ]
            elif facet_by == 'dataset':
                subset = self.combined_data_matrix[
                    (self.combined_data_matrix['dataset'] == facet_val)
                ]
            elif facet_by == 'methylation':
                subset = self.combined_data_matrix
            else:
                raise ValueError(f"Facet by {facet_by} not supported")
                
            if subset_col != '' and subset_val != '':
                subset = subset[subset[subset_col] == subset_val]
            # Add regression line
            sns.regplot(
                data=subset,
                x=actual_age, y=pred_col,
                scatter=False,
                ax=ax,
                color=omics_colors[facet_val] if facet_val in omics_colors.keys() else 'black'
            )
            
            # Add correlation annotations
            
            r, p = pearsonr(subset.dropna(subset=[actual_age, pred_col])[actual_age], subset.dropna(subset=[actual_age, pred_col])[pred_col])
            p_str = f"{p:.2e}".split('e')[0] + " x 10$^{" + f"{p:.2e}".split('e')[1] + "}$"
            ax.annotate(f"r = {r:.2f}\np = {p_str}", 
                    xy=(0.5, 0.11), xycoords=ax.transAxes, 
                    fontsize=8, color='black')
            """rho, pval = spearmanr(subset.dropna(subset=[actual_age, pred_col])[actual_age], subset.dropna(subset=[actual_age, pred_col])[pred_col])
            pval_str = f"{pval:.2e}".split('e')[0] + " × 10$^{" + f"{pval:.2e}".split('e')[1] + "}$"
            ax.annotate(f"ρ = {rho:.2f}\np = {pval_str}", 
                    xy=(0.5, 0.15), xycoords=ax.transAxes, 
                    fontsize=8, color='black')"""
            
            if hue_col == '':
                # Add donor points
                sns.pointplot(
                    data=subset, x=actual_age, y=pred_col,
                    linestyle='none',  native_scale=True, color='black',
                    legend=False,ax=ax,
                    markersize=4,linewidth=1.5,alpha=0.8, rasterized=True
                )
            elif hue_col == 'species':
                sns.scatterplot(
                    data=subset, x=actual_age, y=pred_col, hue=hue_col,
                    palette = {'human': 'black', 'mouse': 'red'},
                    edgecolor = 'black',
                    ax=ax, s=20, alpha=0.6, rasterized=True,
                )
            elif hue_col == 'donor':
                # Add donor points
                sns.pointplot(
                    data=subset, x=actual_age, y=pred_col, hue=hue_col,
                    linestyle='none',  native_scale=True, palette='dark:black',
                    legend=False, ax=ax,
                    markersize=4,linewidth=1.5,alpha=0.8, rasterized=True
                )
            elif hue_col == 'dataset':
                sns.scatterplot(
                    data=subset, x=actual_age, y=pred_col, hue=hue_col,
                    palette = Safe_5.mpl_colors,
                    edgecolor = 'black',
                    ax=ax, s=20, alpha=0.6, rasterized=True
                )
            
            # Set title
            ax.set_title(facet_val)
            
            # x label
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Predicted Age (years)')
            
        # Remove empty subplots if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        return fig, axes
            
            
    def compute_pca(self, n_components: int = 10):
        """
        Perform PCA on the feature matrix (excluding meta columns) and add PC loadings to the data matrix.
        
        Args:
            n_components (int): Number of principal components to compute. Defaults to 10.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Get feature matrix (excluding meta columns)
        feature_matrix = self.combined_data_matrix.drop(columns=self.meta_cols)
        
        # Drop any columns with missing values
        feature_matrix = feature_matrix.dropna(axis=1)
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pc_scores = pca.fit_transform(scaled_features)
        
        # Create DataFrame with PC scores
        pc_df = pd.DataFrame(
            pc_scores,
            index=feature_matrix.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add PC scores to combined data matrix
        for col in pc_df.columns:
            self.combined_data_matrix[col] = pc_df[col]
        
        # Store explained variance ratios
        self.pca_explained_variance_ratio = pca.explained_variance_ratio_
        # Store feature loadings
        self.pca_components = pd.DataFrame(
            pca.components_,
            columns=feature_matrix.columns,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        print(f"Total variance explained by {n_components} PCs: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Add after minmax_scale_within_group method

    def combat_normalize(self, batch_col='dataset'):
        """
        Perform ComBat normalization on non-meta columns using specified batch column.
        Uses pycombat_norm from inmoose package.
        
        Args:
            batch_col (str): Column name containing batch information. Defaults to 'dataset'.
        """
        from inmoose.pycombat import pycombat_norm, pycombat_seq
        
        # Get feature columns (non-meta columns)
        feature_cols = [col for col in self.combined_data_matrix.columns if col not in self.meta_cols]
        
        # Get batch information
        batch = self.combined_data_matrix[batch_col].values
        
        # Extract feature matrix
        feature_matrix = self.combined_data_matrix[feature_cols]
        # convert to int
        #feature_matrix = feature_matrix.astype(int)
        # Perform ComBat normalization
        print(f"Performing ComBat normalization using {batch_col} as batch...", flush=True)
        normalized_matrix = pycombat_norm(feature_matrix.T, batch)
        # Update the combined data matrix with normalized values
        self.combined_data_matrix[feature_cols] = normalized_matrix.T
        
        print("ComBat normalization complete", flush=True)
        
    def compute_nonlinear_pca(
        self, 
        n_components: int = 10, 
        method: str = 'kpca'
        ):
        """
        Perform nonlinear dimensionality reduction on the feature matrix (excluding meta columns) 
        and add component loadings to the data matrix.
        
        Args:
            n_components (int): Number of components to compute. Defaults to 10.
            method (str): Method to use for nonlinear dimensionality reduction.
                Options are 'umap' or 'kpca'. Defaults to 'umap'.
            **kwargs: Additional arguments to pass to UMAP or KernelPCA
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import KernelPCA
        
        # Get feature matrix (excluding meta columns)
        feature_matrix = self.combined_data_matrix.drop(columns=self.meta_cols)
        
        # Drop any columns with missing values
        feature_matrix = feature_matrix.dropna(axis=1)
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        
        if method.lower() == 'kpca':
            reducer = KernelPCA(
                n_components=n_components,
                kernel='rbf',
                random_state=42,
            )
        else:
            raise ValueError(f"Method {method} not supported. Use 'umap' or 'kpca'.")
            
        transformed = reducer.fit_transform(scaled_features)
        
        # Create DataFrame with component scores
        component_df = pd.DataFrame(
            transformed,
            index=feature_matrix.index,
            columns=[f'{method.upper()}{i+1}' for i in range(n_components)]
        )
        
        # Add component scores to combined data matrix
        for col in component_df.columns:
            self.combined_data_matrix[col] = component_df[col]
        
        # Store the reducer for future use
        self.nonlinear_reducer = reducer
        
        if method.lower() == 'kpca':
            # Store explained variance ratios for KPCA
            self.kpca_explained_variance_ratio = reducer.eigenvalues_ / sum(reducer.eigenvalues_)
            print(f"Total variance explained by {n_components} components: {sum(self.kpca_explained_variance_ratio):.2%}")
            
    def apply_model(
        self,
        model,
        feature_names: list,
        result_column: str = 'pred_age'
        ):
        """
        Apply a trained model to the current dataset's features.
        
        Args:
            model: A trained sklearn-like model object
            feature_names (list): List of feature names the model was trained on
            result_column (str): Name of column to store predictions in
            sparse (bool): Whether to convert features to sparse matrix before prediction
                (required for some models like XGBoost). Defaults to False.
        
        Returns:
            None (adds predictions to combined_data_matrix in specified result_column)
        """
        shared_features = self.combined_data_matrix.columns.intersection(feature_names)
        unshared_features = [feature for feature in feature_names if feature not in shared_features]
        print(f"Number of unshared features: {len(unshared_features)}")
        # for any unshared features, add them to the data matrix with the average signal intensity of all shared features
        for feature in unshared_features:
            self.combined_data_matrix[feature] = self.combined_data_matrix[shared_features].mean(axis=1)
        # Get features from combined_data_matrix
        X = self.combined_data_matrix[feature_names]
        
        # Convert to sparse matrix if model is XGBoost
        if isinstance(model, XGBRegressor):
            from scipy.sparse import csr_matrix
            X = csr_matrix(X.values)
            
        # Generate predictions
        predictions = model.predict(X)
        
        # Add predictions to combined_data_matrix
        self.combined_data_matrix[result_column] = predictions
        
        print(f"Added predictions to combined_data_matrix in column: {result_column}")

    def compute_feature_age_correlations(
        self,
        group_cols: list = ['dataset', 'histone_mark'],
        age_col: str = 'age_scaled',
        method: str = 'spearman',
        min_samples: int = 5,
        n_jobs: int = -1
        ) -> pd.DataFrame:
        """
        Calculate correlation between each feature and age within specified groups.
        Parallelized version using multiple threads.
        
        Args:
            group_cols (list): Columns to group by (e.g., ['dataset', 'histone_mark'])
            age_col (str): Column containing age values to correlate against
            method (str): Either 'spearman' or 'mutual_info'. If 'mutual_info',
                returns signed mutual information (MI * sign(spearman correlation))
            min_samples (int): Minimum number of samples required in a group to compute correlation
            n_jobs (int): Number of parallel jobs. -1 means use all available cores.
        
        Returns:
            pd.DataFrame: DataFrame containing correlation values for each feature within each group
        """
        from sklearn.feature_selection import mutual_info_regression
        from scipy.sparse import csr_matrix
        import numpy as np
        from scipy.stats import spearmanr
        from joblib import Parallel, delayed
        
        # Get feature columns (non-meta columns)
        feature_cols = [
            col for col in self.combined_data_matrix.columns 
            if col not in self.meta_cols
        ]
        
        # Create a sparse matrix of features for faster computation
        X = csr_matrix(self.combined_data_matrix[feature_cols].values)
        y = self.combined_data_matrix[age_col].values
        
        # Get group indices
        group_indices = self.combined_data_matrix.groupby(group_cols).indices
        
        def compute_correlations_for_group(group_name, group_idx):
            """Helper function to compute correlations for a single group"""
            if len(group_idx) < min_samples:
                return None
                
            # Extract group data
            X_group = X[group_idx]
            y_group = y[group_idx]
            
            # Convert to dense for feature-wise operations
            X_dense = X_group.toarray()
            
            if method == 'spearman':
                # Vectorized Spearman correlation
                correlations = []
                p_values = []
                n_samples = []
                
                # Use numpy operations for faster computation
                for j in range(X_dense.shape[1]):
                    feature_vals = X_dense[:, j]
                    mask = ~np.isnan(feature_vals)
                    if np.sum(mask) < min_samples:
                        correlations.append(np.nan)
                        p_values.append(np.nan)
                        n_samples.append(0)
                        continue
                        
                    rho, pval = spearmanr(feature_vals[mask], y_group[mask])
                    correlations.append(rho)
                    p_values.append(pval)
                    n_samples.append(np.sum(mask))
                
                results = pd.DataFrame({
                    'feature': feature_cols,
                    'correlation': correlations,
                    'p_value': p_values,
                    'n_samples': n_samples
                })
                
            elif method == 'mutual_info':
                # Compute mutual information for all features at once
                mi_values = mutual_info_regression(X_dense, y_group)
                
                # Get signs from Spearman correlation
                signs = np.array([
                    np.sign(spearmanr(X_dense[:, j], y_group)[0])
                    for j in range(X_dense.shape[1])
                ])
                
                results = pd.DataFrame({
                    'feature': feature_cols,
                    'correlation': mi_values * signs,
                    'p_value': np.nan,
                    'n_samples': len(y_group)
                })
            
            # Add group information
            if isinstance(group_name, tuple):
                for col, val in zip(group_cols, group_name):
                    results[col] = val
            else:
                results[group_cols[0]] = group_name
                
            return results
        
        # Process groups in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_correlations_for_group)(group_name, group_idx)
            for group_name, group_idx in group_indices.items()
        )
        
        # Filter out None results and combine
        results = [r for r in results if r is not None]
        if not results:
            return pd.DataFrame()
            
        results_df = pd.concat(results, ignore_index=True)
        
        # Add multiple testing correction if using Spearman
        if method == 'spearman':
            from statsmodels.stats.multitest import multipletests
            
            # Vectorized FDR correction within groups
            def apply_fdr(group):
                mask = ~group['p_value'].isna()
                if mask.any():
                    _, q_values, _, _ = multipletests(
                        group.loc[mask, 'p_value'],
                        method='fdr_bh'
                    )
                    group.loc[mask, 'q_value'] = q_values
                return group
            
            results_df = results_df.groupby(group_cols, group_keys=False).apply(apply_fdr)
        
        return results_df
    
    def compute_age_feature_overlaps(
        self,
        correlations_df: pd.DataFrame,
        group_cols: list = ['species', 'histone_mark'],
        percentile_threshold: float = .10,
        correlation_col: str = 'correlation',
        min_features: int = 10
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the observed/expected overlap between groups for top age-associated features.
        
        Args:
            correlations_df (pd.DataFrame): Output from compute_feature_age_correlations
            group_cols (list): Columns defining groups to compare
            percentile_threshold (float): Percentage of top/bottom features to consider
            correlation_col (str): Column containing correlation values
            min_features (int): Minimum number of features required in overlap calculation
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing O/E ratios for 
                positive and negative age associations
        """
        # Create group identifier
        correlations_df['group'] = correlations_df[group_cols].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Get unique groups
        groups = correlations_df['group'].unique()
        n_groups = len(groups)
        
        # Initialize matrices for storing results
        pos_oe_matrix = pd.DataFrame(
            np.zeros((n_groups, n_groups)),
            index=groups, columns=groups
        )
        neg_oe_matrix = pd.DataFrame(
            np.zeros((n_groups, n_groups)),
            index=groups, columns=groups
        )
        pos_neg_oe_matrix = pd.DataFrame(
            np.zeros((n_groups, n_groups)),
            index=groups, columns=groups
        )
        neg_pos_oe_matrix = pd.DataFrame(
            np.zeros((n_groups, n_groups)),
            index=groups, columns=groups
        )
        
        # Process each group pair
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i:], i):  # Start from i to avoid duplicates
                # Get features for each group
                features1 = correlations_df[correlations_df['group'] == group1]
                features2 = correlations_df[correlations_df['group'] == group2]
                # Skip if either group has too few features
                if len(features1) < min_features or len(features2) < min_features:
                    pos_oe_matrix.loc[group1, group2] = np.nan
                    pos_oe_matrix.loc[group2, group1] = np.nan
                    neg_oe_matrix.loc[group1, group2] = np.nan
                    neg_oe_matrix.loc[group2, group1] = np.nan
                    pos_neg_oe_matrix.loc[group1, group2] = np.nan
                    pos_neg_oe_matrix.loc[group2, group1] = np.nan
                    neg_pos_oe_matrix.loc[group1, group2] = np.nan
                    neg_pos_oe_matrix.loc[group2, group1] = np.nan
                    continue
                
                
                # Get top percentile_threshold features for each group by ranking
                pos_features1 = set(
                    features1.nlargest(int(len(features1) * percentile_threshold), correlation_col)['feature']
                )
                pos_features2 = set(
                    features2.nlargest(int(len(features2) * percentile_threshold), correlation_col)['feature']
                )
                
                # Get top negative features for each group by ranking
                neg_features1 = set(
                    features1.nsmallest(int(len(features1) * percentile_threshold), correlation_col)['feature']
                )
                neg_features2 = set(
                    features2.nsmallest(int(len(features2) * percentile_threshold), correlation_col)['feature']
                )
                
                # Calculate observed overlaps
                # this is the overlap of the top features
                # divided by the total number of features
                pos_overlap = len(pos_features1.intersection(pos_features2)) / len(set(pos_features1).union(set(pos_features2)))
                pos_neg_overlap = len(pos_features1.intersection(neg_features2)) / len(set(pos_features1).union(set(neg_features2)))
                neg_overlap = len(neg_features1.intersection(neg_features2)) / len(set(neg_features1).union(set(neg_features2)))
                neg_pos_overlap = len(neg_features1.intersection(pos_features2)) / len(set(neg_features1).union(set(pos_features2)))
                
                # the expected overlap is the square of the percentile threshold
                # which we would get if we were to randomly sample features from each group
                expected_overlap = percentile_threshold**2
                
                # Calculate O/E ratios, log transform to center on 0
                pos_oe = np.log2(pos_overlap / expected_overlap)
                neg_oe = np.log2(neg_overlap / expected_overlap)
                pos_neg_oe = np.log2(pos_neg_overlap / expected_overlap)
                neg_pos_oe = np.log2(neg_pos_overlap / expected_overlap)
            
                    
                # calculate pvalues using binomial test
                """from scipy.stats import binom_test
                pos_pval = binom_test(pos_overlap, len(pos_features1), expected_pos)
                pos_neg_pval = binom_test(pos_neg_overlap, len(pos_features1), expected_pos_neg)
                neg_pval = binom_test(neg_overlap, len(neg_features1), expected_neg)
                neg_pos_pval = binom_test(neg_pos_overlap, len(neg_features1), expected_neg_pos)"""
                
                # Store results
                pos_oe_matrix.loc[group1, group2] = pos_oe
                pos_oe_matrix.loc[group2, group1] = pos_oe
                neg_oe_matrix.loc[group1, group2] = neg_oe
                neg_oe_matrix.loc[group2, group1] = neg_oe
                pos_neg_oe_matrix.loc[group1, group2] = pos_neg_oe
                pos_neg_oe_matrix.loc[group2, group1] = pos_neg_oe
                neg_pos_oe_matrix.loc[group1, group2] = neg_pos_oe
                neg_pos_oe_matrix.loc[group2, group1] = neg_pos_oe
                
        # replace negative infinity with 0
        pos_oe_matrix = pos_oe_matrix.replace([np.inf, -np.inf], 0)
        neg_oe_matrix = neg_oe_matrix.replace([np.inf, -np.inf], 0)
        pos_neg_oe_matrix = pos_neg_oe_matrix.replace([np.inf, -np.inf], 0)
        neg_pos_oe_matrix = neg_pos_oe_matrix.replace([np.inf, -np.inf], 0)
        
        
        
        
        
        return pos_oe_matrix, neg_oe_matrix, pos_neg_oe_matrix, neg_pos_oe_matrix
    
    def plot_correlation_radar(
        self,
        correlations_1: dict,
        correlations_2: dict = None,
        figsize: tuple = (4, 4),
        mark_order: list = None,
        label_1: str = 'Set 1',
        label_2: str = 'Set 2',
        color_1: str = TealRose_3.mpl_colors[0],
        color_2: str = TealRose_3.mpl_colors[2],
        value_range: tuple = (0, 1)
        ):
        """
        Create a radar chart comparing correlation values for different marks.
        Can plot either one or two sets of correlations.
        
        Args:
            correlations_1 (dict): Dictionary of mark:correlation pairs for first set
            correlations_2 (dict, optional): Dictionary of mark:correlation pairs for second set
            figsize (tuple): Figure size (width, height)
            mark_order (list, optional): List of marks in desired order around the circle. If not provided,
                                       marks will be ordered alphabetically.
            label_1 (str): Label for first correlation set
            label_2 (str): Label for second correlation set
            color_1 (str): Color for first correlation set
            color_2 (str): Color for second correlation set
            value_range (tuple): Range of values for the radial axis (min, max)
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from math import pi
        
        # Get all marks (categories)
        if mark_order is None:
            categories = sorted(list(correlations_1.keys()))
            if correlations_2 is not None:
                categories = sorted(list(set(categories) | set(correlations_2.keys())))
        else:
            categories = mark_order
        N = len(categories)
        
        # Complete the loop by appending first value
        values_1 = [correlations_1.get(cat, 0) for cat in categories]
        values_1 += values_1[:1]
        
        # Calculate angles for each mark
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot first set of data
        ax.plot(angles, values_1, 'o-', linewidth=1, label=label_1, color=color_1, 
                markerfacecolor=color_1, markeredgecolor='black')
        ax.fill(angles, values_1, alpha=0.25, color=color_1)
        
        # Plot second set if provided
        if correlations_2 is not None:
            values_2 = [correlations_2.get(cat, 0) for cat in categories]
            values_2 += values_2[:1]
            ax.plot(angles, values_2, 'o-', linewidth=1, label=label_2, color=color_2, 
                    markerfacecolor=color_2, markeredgecolor='black')
            ax.fill(angles, values_2, alpha=0.25, color=color_2)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set radial axis range and ticks
        ax.set_ylim(value_range)
        yticks = np.arange(value_range[0], value_range[1] + 0.1, 0.1)
        ax.set_yticks(yticks)
       
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        return fig, ax
    
    def flip_feature_values(self, feature_mark_pairs):
        """
        Invert feature values for specified gene-histone mark pairs.
        
        Args:
            feature_mark_pairs (list): List of [feature, histone_mark] pairs to flip
        
        Returns:
            pd.DataFrame: Data matrix with flipped values for specified pairs
        """
        # Make a copy to avoid modifying original
        data = self.combined_data_matrix.copy()
        
        # Get non-meta columns (features)
        feature_cols = [col for col in data.columns if col not in self.meta_cols]
        
        # For each feature-mark pair
        for feature, mark in feature_mark_pairs:
            if feature in feature_cols:
                # Get mask for samples of this histone mark
                mark_mask = data['histone_mark'] == mark
                
                # Flip values for this feature within the mask
                # First get min/max to maintain scale
                feature_min = data[feature].min()
                feature_max = data[feature].max()
                
                # Invert values within mark samples
                data.loc[mark_mask, feature] = feature_max - (data.loc[mark_mask, feature] - feature_min)
        
        self.combined_data_matrix = data

    def scramble_histone_mark(self, histone_mark: str):
        """
        Scramble feature values for all samples of a specified histone mark.
        Only scrambles non-meta columns and maintains the distribution of values within each sample.
        Uses pandas apply for efficient scrambling.
        
        Args:
            histone_mark (str): Name of histone mark to scramble
        """
        # Get mask for samples of this histone mark
        mark_mask = self.combined_data_matrix['histone_mark'] == histone_mark
        
        # Get non-meta columns (features)
        feature_cols = [col for col in self.combined_data_matrix.columns if col not in self.meta_cols]
        
        # Make a copy of the data to avoid modifying original during scrambling
        scrambled_data = self.combined_data_matrix.copy()
        
        # Define scrambling function for apply
        def scramble_row(row):
            values = row[feature_cols].values
            np.random.shuffle(values)
            row[feature_cols] = values
            return row
        
        # Apply scrambling only to rows with this histone mark
        scrambled_data.loc[mark_mask] = scrambled_data[mark_mask].apply(scramble_row, axis=1)
        
        print(f"Scrambled feature values for {sum(mark_mask)} samples of {histone_mark}")
        self.scrambled_data_matrix = scrambled_data

    # entropy methods
    def _calculate_shannon_entropy(self, row, zero_handling_method='drop'):
        # calculate total sum of row values
        total = row.sum()
        if total == 0:
            return 0
        probabilities = row / total
        EPSILON = 1e-10
        if zero_handling_method == 'drop':
            probabilities = probabilities[probabilities > 0]
            entropy = -np.sum(probabilities * np.log2(probabilities))
        elif zero_handling_method == 'pseudocount':
            smoothed = probabilities + EPSILON
            probabilities = smoothed / smoothed.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
        elif zero_handling_method == 'clip':
            probabilities = np.clip(probabilities, EPSILON, 1.0)
            probabilities = probabilities / probabilities.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
        elif zero_handling_method == 'limit':
            entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))
        return entropy

    def calculate_shannon_entropy(self, input_df: pd.DataFrame = None, zero_handling_method: str = 'drop'):
        # if no input_df provided, use features from combined_data_matrix (non-meta columns)
        if input_df is None:
            features = self.combined_data_matrix.columns.difference(self.meta_cols)
            entropy = self.combined_data_matrix[features].apply(lambda row: self._calculate_shannon_entropy(row, zero_handling_method), axis=1)
        else:
            # if there are negative values, convert them to absolute
            if input_df.min().min() < 0:
                input_df = np.abs(input_df)
            entropy = input_df.apply(lambda row: self._calculate_shannon_entropy(row, zero_handling_method), axis=1)
        return entropy

    def _calculate_cross_entropy(self, row1, row2, zero_handling_method='drop'):
        total1 = row1.sum()
        total2 = row2.sum()
        if total1 == 0 or total2 == 0:
            return 0
        p = row1 / total1
        q = row2 / total2
        EPSILON = 1e-10
        if zero_handling_method == 'drop':
            mask = (p > 0) & (q > 0)
            p = p[mask]
            q = q[mask]
            cross_entropy = -np.sum(p * np.log2(q))
        elif zero_handling_method == 'pseudocount':
            p = p + EPSILON
            q = q + EPSILON
            p = p / p.sum()
            q = q / q.sum()
            cross_entropy = -np.sum(p * np.log2(q))
        elif zero_handling_method == 'clip':
            p = np.clip(p, EPSILON, 1.0)
            q = np.clip(q, EPSILON, 1.0)
            p = p / p.sum()
            q = q / q.sum()
            cross_entropy = -np.sum(p * np.log2(q))
        elif zero_handling_method == 'limit':
            cross_entropy = -np.sum(np.where((p > 0) & (q > 0), p * np.log2(q), 0))
        return cross_entropy

    def calculate_cross_entropy(self, reference_sample: str, zero_handling_method: str = 'pseudocount'):
        features = self.combined_data_matrix.columns.difference(self.meta_cols)
        if reference_sample not in self.combined_data_matrix.index:
            raise ValueError(f"reference sample {reference_sample} not found in combined data")
        ref_dist = self.combined_data_matrix.loc[reference_sample, features]
        cross_entropies = self.combined_data_matrix[features].apply(
            lambda row: self._calculate_cross_entropy(ref_dist, row, zero_handling_method), axis=1
        )
        return cross_entropies