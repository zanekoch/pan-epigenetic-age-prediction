import GEOparse
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
N_CORES = os.cpu_count()


class DatasetDownloader:
    """Class to download a dataset"""
    def __init__(
        self,
        dataset_name:str,
        ) -> None:
        """Constructor for DatasetLoader:
        ### Parameters
        dataset_name : str
            Name of the dataset to download
        ### Returns
        None
        """
        self.dataset_name = dataset_name

    def download_dataset(self):
        if self.dataset_name == "yang_2023":
            self.download_yang_2023()
        elif self.dataset_name == "signal_2024":
            gse = self.download_signal_2024()
            return gse

    def download_signal_2024(self):
        out_dir = os.path.join("../data/signal_2024")
        # create out_dir if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # download the dataset just to get the metadata
        gse = GEOparse.get_GEO(geo="GSE190102", destdir=out_dir, include_data = True)        
        # run the /cellar/users/zkoch/histone_mark_proj/data/signal_2024/download_signal.sh script
        os.system("cd /cellar/users/zkoch/histone_mark_proj/data/signal_2024 && ./download_signal.sh")
        return gse

    def download_yang_2023(self):
        out_dir = os.path.join("../data/yang_2023")
        # create out_dir if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # download the dataset
        gse = GEOparse.get_GEO(geo="GSE185704", destdir=out_dir)
        """# download supplementary files
        gse.download_supplementary_files(directory=out_dir)"""
        # save phenotype data as metadata
        df = gse.phenotype_data
        metadata_df = df[['title', 'characteristics_ch1.0.tissue', 'characteristics_ch1.2.age']].rename(columns = {'characteristics_ch1.0.tissue': 'tissue', 'characteristics_ch1.2.age': 'age_weeks'})
        metadata_df['histone_mark'] = 'H' + metadata_df['title'].str.split('_H').str[-1].str.split('_').str[0]
        metadata_df['age_category'] = metadata_df['title'].str[0]
        # create ages
        metadata_df['age_weeks'] = metadata_df['age_weeks'].str.replace(' weeks', '').astype(float)
        # create age_years column which is 85 weeks if age category is O and 11 weeks if age category is Y
        metadata_df['age_years'] = metadata_df['age_category'].map({'O': 85/52.25, 'Y': 11/52.25})
        # for places where age_weeks is not nan, use that
        metadata_df.loc[metadata_df['age_weeks'].notna(), 'age_years'] = metadata_df.loc[metadata_df['age_weeks'].notna(), 'age_weeks'].astype(int) / 52.25
        # reset index and name is donor
        metadata_df = metadata_df.reset_index().rename(columns = {'index': 'donor'})
        # set index to donor but keep donor as a column
        metadata_df = metadata_df.set_index('donor', drop = False)
        metadata_df.to_parquet(os.path.join(out_dir, "pheno_type_data.parquet"))
        