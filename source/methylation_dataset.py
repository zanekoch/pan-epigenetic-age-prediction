import pandas as pd
import numpy as np
import glob
import os
import pybedtools
import multiprocessing as mp
from tqdm import tqdm
from pybiomart import Dataset
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
import warnings
from pathlib import Path
import pickle
from xgboost import XGBRegressor
# get subproccess
import subprocess

def process_sample(sample_item, regions_bed):
    """
    Process a sample by intersecting with regions and computing mean signal intensity.

    Parameters:
    -----------
    sample_item : tuple
        Tuple containing the sample name and the peaks DataFrame.
    regions_bed : pybedtools.BedTool
        BedTool object for the regions.

    Returns:
    --------
    sample_name : str
        Name of the sample.
    mean_signal : pandas.Series or None
        Series containing the mean signal intensity for each region if there are overlaps,
        otherwise None.
    """
    sample_name, peaks_df = sample_item
    print(f"Processing sample {sample_name}", flush=True)
    # pre-filter peaks_df to only include chromosomes that exist in regions_bed
    try:
        valid_chroms = list(set(regions_bed.to_dataframe()['chrom'].unique()))
        # convert both chromosomes to strings
        peaks_df['chrom'] = peaks_df['chrom'].astype(str)
        valid_chroms = set([str(x) for x in valid_chroms])
        # filter peaks_df to only include chromosomes that exist in regions_bed
        filt_peaks_df = peaks_df[peaks_df['chrom'].isin(valid_chroms)]
        # make sure there are still rows left
        if filt_peaks_df.shape[0] == 0:
            print(f"No valid chromosomes for {sample_name}")
            print(peaks_df.head())
            print(regions_bed.to_dataframe().head())
        else:
            peaks_df = filt_peaks_df
    except:
        print(f"!!!! error in {sample_name}!!!!")
        return sample_name, pd.Series()
    
    # Prepare the peaks bed file
    peaks_df['mf'] = peaks_df['mf'].astype(str)
    peaks_array = peaks_df[['chrom', 'start', 'end', 'mf']].values
    
    # Create BedTool object for peaks
    peaks_bed = pybedtools.BedTool(peaks_array.tolist())
    # Perform intersection
    overlap = peaks_bed.intersect(
        regions_bed, wa=True, wb=True
        )
    # check if there are any overlaps
    if overlap.count() == 0:
        print(f"No overlaps for sample {sample_name}")
        return sample_name, None
    # Convert overlap to DataFrame
    overlap_df = overlap.to_dataframe(
        names=['chrom', 'start', 'end', 'mf',
        'region_chrom', 'region_start', 'region_end', 'region_id']
        )
    # Convert 'mf' back to float
    overlap_df['mf'] = overlap_df['mf'].astype(float)
    # Compute mean score for each region
    mean_signal = overlap_df.groupby('region_id', sort=False)['mf'].mean()
    # round to 3 decimal places
    mean_signal = mean_signal.round(3)
    return sample_name, mean_signal

# for all other datsets besides the blueprint and stubbs
# def read_and_process_bedmethyl_file(fn, regions_bed, bedmethyl_dir):
#     """
#     Read a methylation bed file into a pandas DataFrame.

#     Parameters:
#     -----------
#     fn : str
#         Path to the methylation bed file.
#     regions_bed : pybedtools.BedTool
#         BedTool object for the regions.
    
#     Returns:
#     --------
#     sample_name : str
#         Name of the sample.
#     df : pandas.DataFrame
#         DataFrame containing the methylation data.
#     """
#     sample_name = os.path.basename(fn).split('.')[0]
#     cols = [
#         'chrom',        # chromosome
#         'start',        # start position
#         'end',          # end position
#         'name',         # placeholder
#         'coverage',     # read coverage
#         'strand',       # DNA strand
#         'thick_start',  # same as start
#         'thick_end',    # same as end
#         'rgb',          # methylation color code
#         'block_count',  # same as coverage
#         'mf', # methylation fraction
#         'context',      # sequence context (CG)
#         'sequence',     # actual sequence
#         'region_depth'         # total read depth
#     ]
#     p_fn = fn.replace('.bed.gz', '.parquet')
#     # Read only essential columns for methylation analysis
#     essential_cols = [0, 1, 2, 4, 5, 10, 13]  # chrom, start, end, coverage, strand, meth_percent, depth
#     # check if we can read from parquet
#     try:
#         if os.path.exists(p_fn):
#             df = pd.read_parquet(p_fn)
#         else:
#             df = pd.read_csv(
#                 fn,
#                 sep='\t',
#                 names=cols,
#                 header=None,
#                 compression='gzip',
#                 usecols=essential_cols,
#                 low_memory=False
#                 )
#             # Convert methylation percentage to fraction
#             df['mf'] = df['mf'] / 100.0
#             # write out parquet
#             df.to_parquet(p_fn)
#     except:
#         try:
#             df = pd.read_csv(
#                 fn,
#                 sep='\t',
#                 names=cols,
#                 header=None,
#                 compression='gzip',
#                 usecols=essential_cols,
#                 low_memory=False
#                 )
#             # Convert methylation percentage to fraction
#             df['mf'] = df['mf'] / 100.0
#             # write out parquet
#             df.to_parquet(p_fn)
#         except:
#             print(f"!!!! error reading {sample_name}!!!!")
#             return sample_name, None
    
#     # process the dataframe
#     sample_name, mean_signal = process_sample((sample_name, df), regions_bed)
#     del df
#     # write to bedmethyl_dir
#     out_fn = os.path.join(bedmethyl_dir, f"{sample_name}_mean_signal.parquet")
#     mean_signal.to_frame().to_parquet(out_fn)
#     print(f"wrote to {out_fn}")
#     del mean_signal
#     return sample_name#, mean_signal
    
#For the blueprint and stubbsDNAm
def read_and_process_bedmethyl_file(fn, regions_bed, bedmethyl_dir, sample_name_part=0, split_on='.'):
    """
    Read a methylation bed file into a pandas DataFrame.

    Parameters:
    -----------
    fn : str
        Path to the methylation bed file.
    regions_bed : pybedtools.BedTool
        BedTool object for the regions.
    bedmethyl_dir : str
        Directory to store processed bedmethyl files.
    sample_name_part : int
        The part of the filename to use as the sample name after splitting on split_on.
    split_on : str
        The string to split the filename on to get the sample name.
    
    Returns:
    --------
    sample_name : str
        Name of the sample.
    df : pandas.DataFrame
        DataFrame containing the methylation data.
    """
    sample_name = os.path.basename(fn).split(split_on)[sample_name_part]
    cols = [
        'chrom',        # chromosome
        'start',        # start position
        'end',          # end position
        'mf', # methylation fraction
    ]
    p_fn = fn.replace('.bed.gz', '.parquet')
    # Read only essential columns for methylation analysis
    essential_cols = [0, 1, 2, 3]  # chrom, start, end, meth_percent
    # check if we can read from parquet
    try:
        if os.path.exists(p_fn):
            df = pd.read_parquet(p_fn)
        else:
            df = pd.read_csv(
                fn,
                sep='\t',
                names=cols,
                header=None,
                compression='gzip',
                usecols=essential_cols,
                low_memory=False
                )
            # check if chrom column values start with 'chr' and are strings, if not add chr prefix and convert to string
            """if not all(df['chrom'].str.startswith('chr')):
                df['chrom'] = df['chrom'].astype(str)
                df['chrom'] = 'chr' + df['chrom']"""
            # write out parquet
            df.to_parquet(p_fn)
    except:
        try:
            df = pd.read_csv(
                fn,
                sep='\t',
                names=cols,
                header=None,
                usecols=essential_cols,
                low_memory=False
                )
            # check if chrom column values start with 'chr' and are strings, if not add chr prefix and convert to string
            if not all(df['chrom'].str.startswith('chr')):
                df['chrom'] = df['chrom'].astype(str)
                df['chrom'] = 'chr' + df['chrom']
            # write out parquet
            df.to_parquet(p_fn)
        except:
            print(f"!!!! error reading {sample_name}!!!!")
            return sample_name, None
    
    # process the dataframe
    sample_name, mean_signal = process_sample((sample_name, df), regions_bed)
    del df
    # write to bedmethyl_dir
    out_fn = os.path.join(bedmethyl_dir, f"{sample_name}_mean_signal.parquet")
    mean_signal.to_frame().to_parquet(out_fn)
    print(f"wrote to {out_fn}")
    #del mean_signal
    return sample_name, mean_signal



def starmap_helper(args):
    return read_and_process_bedmethyl_file(args[0], args[1], args[2], args[3], args[4])

class methylationDataset:
    """
    Class to represent a histone mark dataset from ChIP-seq narrowPeak files.
    This class aggregates peak intensity values within genes, promoters, and enhancer regions
    to create a data matrix of samples x genomic locations with mean signal intensity values.
    """

    def __init__(
        self, 
        name: str, 
        species: str, 
        bedmethyl_dir: str, 
        regulatory_features_path: str, 
        gene_locations_path: str,
        metadata_df: pd.DataFrame, 
        n_cores: int = None,
        use_cached_data_matrix: bool = True,
        overwrite_cached_data_matrix: bool = False,
        start_sample_number: int = -1,
        end_sample_number: int = -1,
        sample_name_part: int = 0,
        split_on: str = '.',
        ) -> None:
        """
        Constructor for histoneMarkDataset.

        Parameters:
        -----------
        name : str
            Name of the dataset.
        species : str
            Species name (e.g., 'human', 'mouse').
        bedmethyl_dir : str
            Path to the directory containing bedMethyl files, in (9+2 format (https://www.encodeproject.org/data-standards/wgbs/))
        regulatory_features_path : str
            Path to the GFF3 file containing regulatory features (promoters and enhancers).
        gene_locations_path : str
            Path to the GTF file containing gene locations.
        metadata_df : pandas.DataFrame
            Metadata DataFrame
        n_cores : int, optional
            Number of cores to use for parallel processing. If not specified, os.cpu_count() will be used.
        use_cached_data_matrix : bool, optional
            If True, the data matrix will be read from the cached file if it exists.
        overwrite_cached_data_matrix : bool, optional
            If True, the data matrix will be overwritten if it exists.
        start_sample_number : int, optional
            Start sample number to use, if only processing a subset of samples.
        end_sample_number : int, optional
            End sample number to use, if only processing a subset of samples.
        sample_name_part : int
            The part of the filename to use as the sample name after splitting on split_on.
        split_on : str
            The string to split the filename on to get the sample name.
        """
        self.name = name
        self.species = species
        self.bedmethyl_dir = bedmethyl_dir
        self.regulatory_features_path = regulatory_features_path
        self.gene_locations_path = gene_locations_path
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count()
        self.use_cached_data_matrix = use_cached_data_matrix
        self.overwrite_cached_data_matrix = overwrite_cached_data_matrix
        self.start_sample_number = start_sample_number
        self.end_sample_number = end_sample_number
        self.sample_name_part = sample_name_part
        self.split_on = split_on
        # set metadata
        self.metadata_df = metadata_df
        self.meta_cols = self.metadata_df.columns.tolist()
        self.metadata_df = self.metadata_df[~self.metadata_df.index.duplicated(keep='first')]
        
        # check if there is a cached data matrix
        if self.start_sample_number != -1 and self.end_sample_number != -1:
            data_matrix_fn = os.path.join(self.bedmethyl_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta_{self.start_sample_number}_{self.end_sample_number}.parquet")
        else:
            data_matrix_fn = os.path.join(self.bedmethyl_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta.parquet")
        if os.path.exists(data_matrix_fn) and self.use_cached_data_matrix:
            print(f"Reading cached data matrix from {data_matrix_fn}") 
            self.data_matrix_w_meta = pd.read_parquet(
                data_matrix_fn,
                engine='pyarrow',
                thrift_string_size_limit = 1024*1024*1024
                )
        else:
            # Read in the narrowPeak files and genomic regions (promoters, enhancers, and genes)
            self.regulatory_features = self._read_regulatory_features()
            self.gene_locations = self._read_gene_locations()
            self._create_all_regions()
            self.processed_methyl_dict = self._read_and_process_bedmethyl_files()
            # Get all regions
            self.create_data_matrix()
        
        # create an empty default dict of lists to store models
        self.trained_models = defaultdict(list)
        
    def _read_and_process_bedmethyl_files(self) -> dict:
        """
        Read bedMethyl files into a dictionary.
        """
        # Create BedTool object for regions
        regions_bed = pybedtools.BedTool.from_dataframe(self.all_regions_df)
        # read from parquet if available
        bedmethyl_files = glob.glob(os.path.join(self.bedmethyl_dir, '*.parquet'))
        bedmethyl_files = [f for f in bedmethyl_files if 'mean_signal' not in f]
        bedmethyl_files_csv = glob.glob(os.path.join(self.bedmethyl_dir, '*.bed*'))
        
        """
        # remove files corresponding the samples that are alread in the mean signal files
        bedmethyl_mean_signal_files = glob.glob(os.path.join(self.bedmethyl_dir, '*mean_signal.parquet'))
        mean_signal_samples = [os.path.basename(x).split('_')[0] for x in bedmethyl_mean_signal_files]
        bedmethyl_files = [x for x in bedmethyl_files if os.path.basename(x).split('.')[0] not in mean_signal_samples]
        """
        if len(bedmethyl_files) < len(bedmethyl_files_csv):
            bedmethyl_files = bedmethyl_files_csv
            print("Reading from bed.gz files")
        else:
            print("Reading from parquet files")
        # subset the bedmethyl files if we are only processing a subset of samples
        if self.start_sample_number != -1 and self.end_sample_number != -1:
            bedmethyl_files = bedmethyl_files[self.start_sample_number:self.end_sample_number]
        print(f"Processing {bedmethyl_files}")
        with mp.Pool(self.n_cores) as pool:
            # Use imap with a regular function instead of lambda
            results = list(tqdm(
                pool.imap(
                    starmap_helper,
                    [(fn, regions_bed, self.bedmethyl_dir, self.sample_name_part, self.split_on) for fn in bedmethyl_files]
                ),
                total=len(bedmethyl_files),
                desc="Reading bedMethyl files"
            ))
        processed_methyl_dict = {k: v for k, v in results}
        return processed_methyl_dict
        
    def _read_regulatory_features(self) -> pd.DataFrame:
        """
        Read regulatory features from the GFF3 file.

        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing promoter and enhancer regions.
        """
        print("Reading regulatory features files")
        # Read the GFF3 file
        cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 
                'strand', 'phase', 'attributes']
        df = pd.read_csv(
            self.regulatory_features_path, sep='\t', comment='#', 
            names=cols, header=None, compression='gzip', low_memory=False
            )
        # Filter for promoters and enhancers
        df = df[df['type'].isin(['promoter', 'enhancer'])]
        # Extract ID from attributes
        df['ID'] = df['attributes'].str.extract('ID=([^;]+)')
        # Add region_id
        df['region_id'] = df['type'] + ':' + df['ID']
        # Keep relevant columns
        df = df[['seqid', 'start', 'end', 'region_id']]
        # Rename columns for PyRanges
        df.rename(columns={'seqid': 'Chromosome', 'start': 'Start', 'end': 'End'}, inplace=True)
        return df

    def _read_gene_locations(self) -> pd.DataFrame:
        """
        Read gene locations from the GTF file.

        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing gene regions.
        """
        print("Reading gene locations file")
        # Read the GTF file
        cols = ['seqname', 'source', 'feature', 'start', 'end', 'score', 
                'strand', 'frame', 'attribute']
        df = pd.read_csv(self.gene_locations_path, sep='\t', comment='#', 
                         names=cols, header=None, compression='gzip', low_memory=False)
        # Filter for genes
        df = df[df['feature'] == 'gene']
        # Extract gene_id
        df['gene_id'] = df['attribute'].str.extract('gene_id "([^"]+)"')
        # Add region_id
        df['region_id'] = 'gene:' + df['gene_id']
        # Keep relevant columns
        df = df[['seqname', 'start', 'end', 'region_id']]
        # Rename columns for PyRanges
        df.rename(columns={'seqname': 'Chromosome', 'start': 'Start', 'end': 'End'}, inplace=True)
        return df
    
    def _create_all_regions(self) -> pd.DataFrame:
        """
        Combine regulatory features and gene locations into a single DataFrame.

        Returns:
        --------
        all_regions_df : pandas.DataFrame
            DataFrame containing all genomic regions with region IDs.
        """
        print("Creating all regions dataframe")
        # Get regulatory features dataframe
        reg_df = self.regulatory_features[['Chromosome', 'Start', 'End', 'region_id']]
        # Get gene locations dataframe
        gene_df = self.gene_locations[['Chromosome', 'Start', 'End', 'region_id']]
        # Concatenate
        self.all_regions_df = pd.concat([reg_df, gene_df], ignore_index=True)
        # rename columns to match the bed files
        self.all_regions_df.rename(columns={'Chromosome': 'chrom', 'Start': 'start', 'End': 'end', 'region_id': 'region_id'}, inplace=True)
        self.all_regions_df['region_id'] = self.all_regions_df['region_id'].astype(str)
        # filter out non autosomes
        self.all_regions_df = self.all_regions_df[self.all_regions_df['chrom'].isin([str(i) for i in range(1, 23)] + ['X', 'Y'])]
        # add chr prefix
        self.all_regions_df['chrom'] = 'chr' + self.all_regions_df['chrom']
        
    def create_data_matrix(self) -> pd.DataFrame:
        """
        Create the data matrix of samples x genomic locations with mean signal intensity.

        Parameters:
        -----------
        results : dict
            Dictionary containing sample names and mean signal intensity. From running read_and_process_bedmethyl_file.

        Returns:
        --------
        data_matrix_w_meta : pandas.DataFrame
            DataFrame containing the aggregated data with metadata.
        """
        print("Creating data matrix")
        region_ids = self.all_regions_df['region_id'].unique()
        # Get samples
        samples = list(self.processed_methyl_dict.keys())
        # Create an empty dataframe to store the results
        self.data_matrix = pd.DataFrame(index=samples, columns=region_ids)
            
        # populate the data matrix
        for sample_name, mean_signal in tqdm(
            self.processed_methyl_dict.items(), desc="Populating data matrix"
            ):
            if mean_signal is not None:
                self.data_matrix.loc[sample_name, mean_signal.index] = mean_signal
            else:
                self.data_matrix.loc[sample_name, :] = 0

        # convert all columns to float
        self.data_matrix = self.data_matrix.astype(float)
        # fill nans with 0
        """self.data_matrix = self.data_matrix.fillna(0)"""
        # convert both indices to strings
        self.data_matrix.index = self.data_matrix.index.astype(str)
        self.metadata_df.index = self.metadata_df.index.astype(str)
        # merge the data matrix with the metadata
        self.data_matrix_w_meta = self.data_matrix.merge(
            self.metadata_df, left_index=True, right_index=True, how = 'left'
            ).dropna(subset = ['age_years', 'donor'])
        
        # cache the data matrix
        if self.overwrite_cached_data_matrix:
            os.makedirs(os.path.join(self.bedmethyl_dir, 'data_matrix'), exist_ok=True)
            if self.start_sample_number != -1 and self.end_sample_number != -1:
                data_matrix_fn = os.path.join(self.bedmethyl_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta_{self.start_sample_number}_{self.end_sample_number}.parquet")
            else:
                data_matrix_fn = os.path.join(self.bedmethyl_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta.parquet")
            self.data_matrix_w_meta.to_parquet(
                data_matrix_fn
            )
            
    def predict(
        self, 
        target_column: str, 
        model, 
        result_column: str, 
        n_folds: int = 5,
        features: list = None,
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
        """
        # Prepare data
        y = self.data_matrix_w_meta[target_column]
        X = self.data_matrix_w_meta        

        # subset the data matrix to the specified features
        if features is not None:
            X = X[features + self.meta_cols]

        # drop the meta columns, except for 'donor'
        # so that we can keep samples from the same donors in the same fold
        meta_cols_to_drop = [col for col in self.meta_cols if col != 'donor']

        print(f"Training model", flush=True)
        predictions = self._do_kfold_cv(
            model = model, 
            X = X.drop(columns=meta_cols_to_drop), 
            y = X[target_column], 
            n_folds = n_folds, 
            result_column = result_column
            )
        
        # update the data matrix with the predictions
        self.data_matrix_w_meta[result_column] = self.data_matrix_w_meta.index.map(predictions)
            
        # Calculate and print the mean squared error
        mse = mean_squared_error(y, self.data_matrix_w_meta[result_column])
        r, p = pearsonr(y, self.data_matrix_w_meta[result_column])
        rho, p_s = spearmanr(y, self.data_matrix_w_meta[result_column])
        print(f"Mean Squared Error for {target_column}: {mse}")
        print(f"Pearson's r for {target_column}: {r}, p = {p}")
        print(f"Spearman's rho for {target_column}: {rho}, p = {p_s}")
        
    def _do_kfold_cv(self, model, X, y, n_folds, result_column) -> dict:
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
        
        Returns:
        --------
        y_pred_dict : dict
            A dictionary with the index of the test set as the key and the predicted value as the value.
        """
        self.trained_models[result_column] = []
        # Initialize StratifiedGroupKFold, balancing by age and grouping by donor
        sgf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        groups = X['donor']
        # convert age to decades
        classes = (y / 10).astype(int)
        # save the predicted values with the index
        y_pred_dict = {}
        # Perform cross-validation
        for _, (train_index, test_index) in enumerate(sgf.split(X, classes, groups=groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, _ = y.iloc[train_index], y.iloc[test_index]
            
            warnings.filterwarnings("ignore")
            # remove donor from the features
            X_train = X_train.drop(columns=['donor'])
            X_test = X_test.drop(columns=['donor'])
            # Train the model
            model = model.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test)
            
            # Store the predictions
            y_pred_dict.update({X_test.index[i]: y_pred[i] for i in range(len(y_pred))})
            
            # Save the trained model
            self.trained_models[result_column].append(model)
            
        return y_pred_dict
    
    def predict_on_cluster(
        self,
        output_dir,
        result_column,
        target_column="age_years",
        features=None,
        time="24:00:00",
        mem_gb=32,
        cpus=4,
        jobname = 'mthclck'
    ):
        """
        Submit prediction job to Slurm cluster as a class method
        
        Args:
            output_dir: Directory to save results and models
            result_column: Name for prediction column
            target_column: Column to predict
            features: List of features to use (defaults to all)
            time: Time limit for job (HH:MM:SS)
            mem_gb: Memory in GB
            cpus: Number of CPUs to request
            jobname: name of job
        """

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create python script for prediction
        script_content = f"""
import pickle
import os
from sklearn.linear_model import ElasticNetCV
import sys
source_path = os.path.abspath(os.path.join('/cellar/users/zkoch/histone_mark_proj'))
if source_path not in sys.path:
    sys.path.append(os.path.join(source_path, 'source'))
from histone_mark_dataset import histoneMarkDataset
from methylation_dataset import methylationDataset
    
# Load data object
with open('{output_dir}/data_object.pkl', 'rb') as f:
    data_obj = pickle.load(f)

# Run prediction
results = data_obj.predict(
    target_column='{target_column}',
    model=ElasticNetCV(n_jobs={cpus}, random_state=42, verbose=1),
    result_column='{result_column}',
    n_folds=5,
    features={features} if {features} else data_obj.data_matrix_w_meta.columns.tolist(),
)

# Save predictions, which are in the data_matrix_w_meta dataframe
data_obj.data_matrix_w_meta['{result_column}'].to_csv('{output_dir}/predictions.csv')

# Save models
with open('{output_dir}/models.pkl', 'wb') as f:
    pickle.dump(data_obj.trained_models, f)
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
        
    def read_cluster_predictions(
        self,
        out_dir: str
        ):
        """
        Read in predictions from cluster jobs.

        Args:
            out_dir (str): Directory containing prediction results
        """
        # Read predictions
        predictions = pd.read_csv(os.path.join(out_dir, 'predictions.csv'), index_col=0)
        # Read models
        self.trained_models = pd.read_pickle(os.path.join(out_dir, 'models.pkl'))
            
        # Update data matrix with predictions
        pred_name = out_dir.split('/')[-1]
        self.data_matrix_w_meta[pred_name] = predictions
        
        print(f"Added predictions to data_matrix_w_meta in {pred_name}")
        print("read in trained models")
        
    