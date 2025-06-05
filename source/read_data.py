import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from histone_mark_dataset import histoneMarkDataset
from methylation_dataset import methylationDataset
N_CORES = os.cpu_count()


class DatasetLoader:
    """Class to load a dataset"""
    def __init__(
        self,
        dataset_name:str,
        load_histone: bool = True,
        load_methyl: bool = True
        ) -> None:
        """Constructor for DatasetLoader:
        Optionally choose which datasets (in case you don't want to wait for methylation for example)
        ### Parameters
        dataset_name : str
            Name of the dataset to load
        load_histone : bool
            Whether to load histone data (if available)
        load_methyl : bool
            Whether to load methylation data (if available)
        ### Returns
        None
        """
        self.dataset_name = dataset_name
        self.load_histone = load_histone
        self.load_methyl = load_methyl

    def load_dataset(
        self,
        ):
        """Load the dataset based on the dataset_name
        ### Returns
        A list of dataset objects or a single dataset object : list | 
            The contents of this list depends on the dataset
        """
        print(f"Loading dataset: {self.dataset_name}")
        if self.dataset_name == "encode-human":
            dataset_list = self.load_encode_human()
        elif self.dataset_name == 'encode-mouse':
            dataset_list = self.load_encode_mouse()
        elif self.dataset_name == 'blueprint-human':
            dataset_list = self.load_blueprint_human()
        elif self.dataset_name == 'yang_2023':
            dataset_list = self.load_yang_2023()
        elif self.dataset_name == 'signal_2024':
            dataset_list = self.load_signal_2024()
        elif self.dataset_name == 'stubbs_2017':
            dataset_list = self.load_stubbs()
        elif self.dataset_name == 'hillje_2022':
            dataset_list = self.load_hillje_2022()
        elif self.dataset_name == 'CEEHRC':
            dataset_list = self.load_CEEHRC()
        elif self.dataset_name == 'petkovich':
            dataset_list = self.load_petkovich()
        elif self.dataset_name == 'meer':
            dataset_list = self.load_meer()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found")
        return dataset_list
    
    def load_meer(self):
        """Load the Meer methylation dataset
        
        Returns
        -------
        methylationDataset
            The loaded Meer methylation dataset
        """
        metadata_df = pd.read_csv("/cellar/users/zkoch/histone_mark_proj/data/meer_2018/metadata_final.csv", index_col=0)
        
        meer_methyl = methylationDataset(
            name = "meer",
            species = "mouse",
            bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/meer_2018/bed_methyl",
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz",
            metadata_df = metadata_df,
            n_cores = N_CORES,
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            sample_name_part = 0,
            split_on = '_',
        )
        return meer_methyl
    
    def load_petkovich(self):
        """Load the Petkovich methylation dataset
        
        Returns
        -------
        methylationDataset
            The loaded Petkovich methylation dataset
        """
        metadata_df = pd.read_csv("/cellar/users/zkoch/histone_mark_proj/data/petkovich_2017/metadata_final.csv", index_col=0)
        
        petkovich_methyl = methylationDataset(
            name = "petkovich",
            species = "mouse",
            bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/petkovich_2017/bed_methyl",
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz",
            metadata_df = metadata_df,
            n_cores = N_CORES,
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            sample_name_part = 0,
            split_on = '_',
        )
        return petkovich_methyl
    
    def load_CEEHRC(self):
        """Load the CEEHRC dataset
        ### Returns
        List[histoneMarkDataset, methylationDataset]
        """
        metadata_df = pd.read_csv("/cellar/users/zkoch/histone_mark_proj/data/CEEHRC/metadata_final.csv", index_col=0)
        

        ceehrc_methyl = methylationDataset(
            name = "CEEHRC",
            species = "human",
            bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/CEEHRC/methyl_bed",
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz",
            metadata_df = metadata_df,
            n_cores = N_CORES,
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            sample_name_part = 2,
            split_on = '.',
            )
        ceehrc_methyl.data_matrix_w_meta.rename(columns = {'mark': 'histone_mark'}, inplace = True)
        ceehrc_methyl.meta_cols.remove('mark')
        ceehrc_methyl.meta_cols.append('histone_mark')

        ceehrc_hmark = histoneMarkDataset(
            name = "ceehrc_hmark",
            species = "human",
            narrowpeak_dir = "/cellar/users/zkoch/histone_mark_proj/data/CEEHRC/bed_peaks",
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz",
            metadata_df = metadata_df,
            n_cores = N_CORES,
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            )
        ceehrc_hmark.meta_cols.remove('mark')
        ceehrc_hmark.meta_cols.append('histone_mark')
        return [ceehrc_methyl, ceehrc_hmark]
            
    
    def load_hillje_2022(self):
        """Load the Hillje et al. 2022 dataset
        ### Returns
        List[histoneMarkDataset]
        """
        # 12 min for all at 16 cores
        hillje = histoneMarkDataset(
            name = "hillje",
            species = "mouse",
            narrowpeak_dir = "/cellar/users/zkoch/histone_mark_proj/data/hillje_2022/broad_peaks",
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
            metadata_df = pd.read_csv("/cellar/users/zkoch/histone_mark_proj/data/hillje_2022/metadata.csv", index_col=0),
            n_cores = N_CORES,
            split_on = '_',
            overwrite_cached_data_matrix = False,
            use_cached_data_matrix = True
        )
        return hillje
    
    def load_stubbs(self):
        """Load the Stubbs methylation dataset
        ### Returns
        methylationDataset
        """
        metadata_df = pd.read_csv(
                    '/cellar/users/zkoch/histone_mark_proj/data/stubbs_2017/metadata.csv',
                    index_col = 0
                    )
        # Initialize methylation dataset
        stubbs_methyl = methylationDataset(
            name="stubbs_methyl",
            species="mouse",
            bedmethyl_dir='/cellar/users/zkoch/histone_mark_proj/data/stubbs_2017',
            regulatory_features_path="/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path="/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz",
            metadata_df=metadata_df,
            n_cores=N_CORES,
            use_cached_data_matrix=True,
            overwrite_cached_data_matrix=False
        )
        return stubbs_methyl
            
    def load_signal_2024(self):
        """Load the Signal et al. 2024 dataset
        ### Returns
        List[histoneMarkDataset]
        """
        signal = histoneMarkDataset(
            name = 'signal_2024',
            species = 'mouse',
            narrowpeak_dir = '/cellar/users/zkoch/histone_mark_proj/data/signal_2024/bed_peaks',
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
            metadata_df = pd.read_parquet('/cellar/users/zkoch/histone_mark_proj/data/signal_2024/pheno_type_data.parquet'),
            n_cores = 16,
            sample_name_part = 0,
            split_on = '.',
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            )
        # add tissue column
        signal.data_matrix_w_meta['tissue'] = 'brain'
        return signal
    
    def load_yang_2023(self):
        """Load the MsCauley 2021 dataset
        ### Returns
        List[histoneMarkDataset]
        """
        yang = histoneMarkDataset(
            name = 'yang_2023',
            species = 'mouse',
            narrowpeak_dir = '/cellar/users/zkoch/histone_mark_proj/data/yang_2023/bed_peaks',
            regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
            gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
            metadata_df = pd.read_parquet('/cellar/users/zkoch/histone_mark_proj/data/yang_2023/pheno_type_data.parquet'),
            n_cores = 16,
            use_cached_data_matrix = True,
            overwrite_cached_data_matrix = False,
            sample_name_part = 0,
            split_on = '_'
            )
        return yang
            
    def load_encode_human(self):
        """Load the ENCODE human dataset
        ### Returns
        List[histoneMarkDataset, methylationDataset]
        """
        if self.load_methyl:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"human_methyl_metadata_fixed.parquet")).set_index('File accession')
            methyl_human_encode = methylationDataset(
                name = "methyl_human_encode",
                species = "homo_sapiens",
                bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/human_bed_files/methyl",
                regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
                gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz", 
                metadata_df = metadata_df,
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False
                )
        if self.load_histone:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"human_histone_metadata_fixed.parquet")).set_index('File accession')
            human_encode_hmark = histoneMarkDataset(
                name = "human_encode_hmark",
                species = "homo_sapiens",
                narrowpeak_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/human_bed_files/histone",
                regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
                gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz", 
                metadata_df = metadata_df,
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False,
                )
        if self.load_methyl and self.load_histone:
            return [methyl_human_encode, human_encode_hmark]
        elif self.load_methyl:
            return methyl_human_encode
        else:
            return human_encode_hmark
    
    def load_encode_mouse(self):
        """Load the ENCODE mouse dataset
        ### Returns
        List[methylationDataset, histoneMarkDataset]
        """
        if self.load_methyl:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"mouse_methyl_metadata_fixed.parquet")).set_index('File accession')
            mouse_encode_methyl = methylationDataset(
                name = "mouse_encode_methyl",
                species = "mus_musculus",
                bedmethyl_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/mouse_bed_files/methyl",
                regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
                gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
                metadata_df = metadata_df,
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False,
                )
        if self.load_histone:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/encode/metadata', f"mouse_histone_metadata_fixed.parquet")).set_index('File accession')
            mouse_encode_hmark = histoneMarkDataset(
                name = "mouse_encode_hmark",
                species = "mus_musculus",
                narrowpeak_dir = "/cellar/users/zkoch/histone_mark_proj/data/encode/mouse_bed_files/histone",
                regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/mus_musculus.GRCm38.Regulatory_Build.regulatory_features.20180516.gff.gz",
                gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Mus_musculus.GRCm38.102.gtf.gz", 
                metadata_df = metadata_df,
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False,
                )
        if self.load_methyl and self.load_histone:
            return [mouse_encode_methyl, mouse_encode_hmark]
        elif self.load_methyl:
            return mouse_encode_methyl
        else:
            return mouse_encode_hmark
    
    def load_blueprint_human(self):
        """Load the Blueprint human dataset
        ### Returns
        List[histoneMarkDataset]
        """
        if self.load_histone:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/blueprint/metadata', f"blueprint_metadata_final.parquet")).set_index('EXPERIMENT_ID')
            metadata_df.rename(columns = {'DONOR_AGE': 'age_years', 'SAMPLE_NAME': 'donor', 'TISSUE_TYPE': 'tissue'}, inplace = True)
            # Convert age ranges to midpoints
            metadata_df['age_years'] = metadata_df['age_years'].apply(self.map_age_range_to_midpoint)
            # dropna
            metadata_df = metadata_df.dropna(subset = ['age_years'])
            # rename 
            human_blueprint_hmark = histoneMarkDataset(
                name = "human_blueprint_hmark",
                species = "homo_sapiens",
                narrowpeak_dir = "/cellar/users/zkoch/histone_mark_proj/data/blueprint/histone",
                regulatory_features_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.regulatory_features.v112.gff3.gz",
                gene_locations_path = "/cellar/users/zkoch/histone_mark_proj/data/ensembl_reference/Homo_sapiens.GRCh38.112.gtf.gz", 
                metadata_df = metadata_df,
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False,
                sample_name_part = 1,
            )
            human_blueprint_hmark.data_matrix_w_meta.rename(columns = {'TISSUE_TYPE': 'tissue_type'}, inplace = True)
            human_blueprint_hmark.data_matrix_w_meta.rename(columns = { 'tissue_type':'tissue'}, inplace = True)
            human_blueprint_hmark.meta_cols = human_blueprint_hmark.data_matrix_w_meta.iloc[:, -11:].columns.tolist()
        if self.load_methyl:
            metadata_df = pd.read_parquet(os.path.join('/cellar/users/zkoch/histone_mark_proj/data/blueprint/metadata', f"blueprint_metadata_final.parquet")).set_index('EXPERIMENT_ID')
            metadata_df.rename(columns = {'DONOR_AGE': 'age_years', 'SAMPLE_NAME': 'donor', 'TISSUE_TYPE': 'tissue'}, inplace = True)
            # Convert age ranges to midpoints
            metadata_df['age_years'] = metadata_df['age_years'].apply(self.map_age_range_to_midpoint)
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
                n_cores = N_CORES,
                use_cached_data_matrix = True,
                overwrite_cached_data_matrix = False
                )
            human_blueprint_methyl.data_matrix_w_meta.rename(columns = {'TISSUE_TYPE': 'tissue_type'}, inplace = True)
            human_blueprint_methyl.data_matrix_w_meta.rename(columns = { 'tissue_type':'tissue'}, inplace = True)
            human_blueprint_methyl.meta_cols = human_blueprint_methyl.data_matrix_w_meta.iloc[:, -11:].columns.tolist()
        if self.load_methyl and self.load_histone:
            return [human_blueprint_methyl, human_blueprint_hmark]
        elif self.load_methyl:
            return human_blueprint_methyl
        else:
            return human_blueprint_hmark
        
    def map_age_range_to_midpoint(self, age_range):
        if age_range == '-' or pd.isna(age_range):
            return np.nan
        elif age_range == '90+':
            return 92.5 # assume a little higher than 90, since most probably are
        elif ' - ' in str(age_range):
            start, end = map(float, age_range.split(' - '))
            return (start + end) / 2
        else:
            return float(age_range)  # For single numbers
        
        
    def read_soft(
        self,
        fn : str
        ) -> pd.DataFrame:
        """Read metadata from .soft file
        ### Parameters
        fn : str
            Path to .soft file
        ### Returns
        metadata : pd.DataFrame
            Metadata dataframe
        """
        # create defaultDict filled with empty lists
        metadata = defaultdict(list)
        
        with open(fn, "r") as f:
            # iterate over lines
            lines = f.readlines()
            for i in range(len(lines)):
                # until we find a new sample, keep going
                if lines[i].startswith("^SAMPLE"):
                    # once we find a new sample, add it as a key
                    sample_name = lines[i].strip("^SAMPLE = ").strip()
                    metadata['sample'].append(sample_name)
                    # then continue looping over subsequent lines
                    not_next_sample = True
                    while not_next_sample:
                        i += 1
                        # if we find a new sample, break
                        if lines[i].startswith("^SAMPLE") or i == len(lines)-1:
                            i -= 1
                            not_next_sample = False
                            break
                        # otherwise, add things we want
                        else:
                            """elif lines[i].startswith("!Sample_description"):
                                    metadata['sample_name'].append(
                                        lines[i].strip("!Sample_description = ").strip()
                                        )"""
                            if lines[i].startswith("!Sample_geo_accession"):
                                metadata['sample_geo_accession'].append(
                                    lines[i].strip("!Sample_geo_accession = ").strip()
                                    )
                            if lines[i].startswith("!Sample_title"):
                                metadata['sample_title'].append(
                                    lines[i].strip("!Sample_title = ").strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = Sex:"):
                                # why have to do this way?
                                to_remove = "!Sample_characteristics_ch1 = Sex: " 
                                metadata['sex'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = gender:"):
                                # why have to do this way?
                                to_remove = "!Sample_characteristics_ch1 = gender: " 
                                metadata['sex'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = genotype:"):
                                # why have to do this way?
                                to_remove = "!Sample_characteristics_ch1 = genotype: " 
                                metadata['intervention'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = age:"):
                                to_remove = "!Sample_characteristics_ch1 = age: " 
                                metadata['age'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = treatment"):
                                to_remove = "!Sample_characteristics_ch1 = treatment: "
                                metadata['treatment'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = intervention duration"):
                                to_remove = "!Sample_characteristics_ch1 = intervention duration: "
                                metadata['intervention_duration'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = intervention"):
                                to_remove = "!Sample_characteristics_ch1 = intervention: "
                                metadata['intervention'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = dose"):
                                to_remove = "!Sample_characteristics_ch1 = dose: "
                                metadata['dose'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_characteristics_ch1 = Sex"):
                                to_remove = "!Sample_characteristics_ch1 = Sex: "
                                metadata['sex'].append(
                                    lines[i][len(to_remove):].strip()
                                    )      
                            elif lines[i].startswith("!Sample_characteristics_ch1 = tissue"):
                                to_remove = "!Sample_characteristics_ch1 = tissue: "
                                metadata['tissue'].append(
                                    lines[i][len(to_remove):].strip()
                                    )  
                            elif lines[i].startswith("!Sample_characteristics_ch1 = Stage"):
                                to_remove = "!Sample_characteristics_ch1 = Stage: "
                                metadata['condition'].append(
                                    lines[i][len(to_remove):].strip()
                                    )  
                            elif lines[i].startswith("!Sample_data_processing = "):
                                to_remove = "!Sample_data_processing = "
                                metadata['data_processing'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_extract_protocol_ch1 = "):
                                to_remove = "!Sample_extract_protocol_ch1 = "
                                if "12 male mice" in lines[i]:
                                    continue
                                metadata['sample_extraction_protocol'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_organism_ch1 = "):
                                to_remove = "!Sample_organism_ch1 = "
                                metadata['organism'].append(
                                    lines[i][len(to_remove):].strip()
                                    )
                            elif lines[i].startswith("!Sample_source_name_ch1 = "):
                                to_remove = "!Sample_source_name_ch1 = "
                                metadata['tissue'].append(
                                    lines[i][len(to_remove):].strip()
                                    )                                
        # turn into dataframe
        # if the lists are not the same lenght pad with nans
        max_len = max([len(x) for x in metadata.values()])
        for key in metadata.keys():
            if len(metadata[key]) < max_len:
                metadata[key] = metadata[key] + [np.nan] * (max_len - len(metadata[key]))
        metadata = pd.DataFrame(metadata)
        
        if self.dataset_name == 'cao_2024':
            # drop rows that are nan in sample column
            metadata.dropna(subset=['sample'], inplace=True)
        else:
            # drop rows that have nan
            metadata.dropna(how='any', inplace=True, axis = 0)
        return metadata
    

    
