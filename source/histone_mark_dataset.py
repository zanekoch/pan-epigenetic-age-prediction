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
import os
from pathlib import Path
import pickle
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

def read_narrowpeak_file(fn, sample_name_part = 0, split_on = '.'):
    """
    Read a narrowPeak file into a pandas DataFrame.

    Parameters:
    -----------
    fn : str
        Path to the narrowPeak file.
    sample_name_part : int
        The part of the filename to use as the sample name. 0 for encode, 1 for blueprint.
    split_on : str
        The string to split the filename on to get the sample name. Defaults to '.' (for encode and blueprint), use '_' for yang.

    Returns:
    --------
    sample_name : str
        Name of the sample.
    df : pandas.DataFrame
        DataFrame containing the narrowPeak data.
    """
    sample_name = os.path.basename(fn).split(split_on)[sample_name_part]
    cols = ['chrom', 'start', 'end', 'name', 'score']
    if fn.endswith('.parquet'):
        df = pd.read_parquet(fn)
    else:
        df = pd.read_csv(fn, sep='\t', names=cols, header=None, compression='gzip', usecols=[0,1,2,3,4], low_memory=False)
        # write out parquet
        df.to_parquet(fn.replace('.bed.gz', '.parquet'))
    return sample_name, df

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
    # Prepare the peaks bed file
    peaks_bed_df = peaks_df[['chrom', 'start', 'end', 'score']].copy()
    # Convert 'signalValue' to string because BedTool expects string columns
    peaks_bed_df['score'] = peaks_bed_df['score'].astype(str)
    # Create BedTool object for peaks
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks_bed_df)
    # Perform intersection
    overlap = peaks_bed.intersect(regions_bed, wa=True, wb=True)
    if overlap.count() == 0:
        print(f"No overlaps for sample {sample_name}")
        return sample_name, None
    # Convert overlap to DataFrame
    overlap_df = overlap.to_dataframe(names=[
        'chrom', 'start', 'end', 'score',
        'region_chrom', 'region_start', 'region_end', 'region_id'])
    # Convert 'score' back to float
    overlap_df['score'] = overlap_df['score'].astype(float)
    # Compute mean score for each region
    mean_signal = overlap_df.groupby('region_id')['score'].mean()
    return sample_name, mean_signal


def convert_ensembl_ids(ids, from_species, to_species, id_type='gene'):
    """
    Convert Ensembl IDs between human and mouse.
    
    :param ids: List of Ensembl IDs to convert
    :param from_species: Source species ('human' or 'mouse')
    :param to_species: Target species ('human' or 'mouse')
    :param id_type: Type of ID to convert ('gene' or 'regulatory')
    :return: Dictionary mapping input IDs to converted IDs
    """
    # Validate input parameters
    if from_species not in ['human', 'mouse'] or to_species not in ['human', 'mouse']:
        raise ValueError("Species must be either 'human' or 'mouse'")
    if from_species == to_species:
        raise ValueError("Source and target species must be different")
    if id_type not in ['gene', 'regulatory']:
        raise ValueError("ID type must be either 'gene' or 'regulatory'")

    # Set up the correct datasets and attributes
    species_dataset = {'human': 'hsapiens_gene_ensembl', 'mouse': 'mmusculus_gene_ensembl'}
    from_dataset = Dataset(name=species_dataset[from_species], host='http://www.ensembl.org')
    to_dataset = Dataset(name=species_dataset[to_species], host='http://www.ensembl.org')

    if id_type == 'gene':
        from_attr = f'ensembl_gene_id'
        to_attr = f'mmusculus_homolog_ensembl_gene'
    else:  # regulatory
        from_attr = f'ensembl_regulatory_id'
        to_attr = f'mmusculus_homolog_regulatory_feature'

    # Perform the conversion
    result = from_dataset.query(attributes=[from_attr, to_attr])
    
    # Create a dictionary of the results
    #conversion_dict = dict(zip(result[from_attr], result[to_attr]))
    
    return result

# Example usage:
# human_ids = ['ENSG00000139618', 'ENSG00000141510']
# mouse_ids = convert_ensembl_ids(human_ids, 'human', 'mouse', 'gene')
# print(mouse_ids)

# regulatory_ids = ['ENSR00000000001', 'ENSR00000000002']
# mouse_regulatory_ids = convert_ensembl_ids(regulatory_ids, 'human', 'mouse', 'regulatory')
# print(mouse_regulatory_ids)

class histoneMarkDataset:
    """
    Class to represent a histone mark dataset from ChIP-seq narrowPeak files.
    This class aggregates peak intensity values within genes, promoters, and enhancer regions
    to create a data matrix of samples x genomic locations with mean signal intensity values.
    """

    def __init__(
        self, 
        name: str, 
        species: str, 
        narrowpeak_dir: str, 
        regulatory_features_path: str, 
        gene_locations_path: str,
        metadata_df: pd.DataFrame, 
        n_cores: int = None,
        use_cached_data_matrix: bool = True,
        overwrite_cached_data_matrix: bool = False,
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
        narrowpeak_dir : str
            Path to the directory containing narrowPeak bed.gz files.
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
        sample_name_part : int
            The part of the filename to use as the sample name. 0 for encode, 1 for blueprint.
        split_on : str
            The string to split the filename on to get the sample name. Defaults to '.' (for encode and blueprint), use '_' for yang.
        """
        self.name = name
        self.species = species
        self.narrowpeak_dir = narrowpeak_dir
        self.regulatory_features_path = regulatory_features_path
        self.gene_locations_path = gene_locations_path
        self.n_cores = n_cores if n_cores is not None else mp.cpu_count()
        self.use_cached_data_matrix = use_cached_data_matrix
        self.overwrite_cached_data_matrix = overwrite_cached_data_matrix
        self.sample_name_part = sample_name_part
        self.split_on = split_on
        
        # set metadata
        self.metadata_df = metadata_df
        self.meta_cols = self.metadata_df.columns.tolist()
        self.metadata_df = self.metadata_df[~self.metadata_df.index.duplicated(keep='first')]
        
        # check if there is a cached data matrix
        data_matrix_fn = os.path.join(
            self.narrowpeak_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta.parquet"
            )
        if os.path.exists(data_matrix_fn) and self.use_cached_data_matrix:
            print(f"Reading cached data matrix from {data_matrix_fn}")
            self.data_matrix_w_meta = pd.read_parquet(
                data_matrix_fn,
                engine='pyarrow',
                thrift_string_size_limit = 1024*1024*1024
                )
        else:
            # Read in the narrowPeak files and genomic regions (promoters, enhancers, and genes)
            self.hmark_dict = self._read_narrowpeak_files()
            self.regulatory_features = self._read_regulatory_features()
            self.gene_locations = self._read_gene_locations()
            # Get all regions
            self._create_all_regions()
            self.create_data_matrix()
            
        # drop samples with nans in age_years
        self.data_matrix_w_meta.dropna(inplace = True, subset = ['age_years'])
        # create an empty default dict of lists to store models
        self.trained_models = defaultdict(list)


    def _read_narrowpeak_files(self) -> dict:
        """
        Read narrowPeak files from the specified directory.

        Returns:
        --------
        samples : dict
            Dictionary with sample names as keys and pandas DataFrames as values.
        """
        # read from parquet if available
        narrowpeak_files = glob.glob(os.path.join(self.narrowpeak_dir, '*.parquet'))
        narrowpeak_files_csv = glob.glob(os.path.join(self.narrowpeak_dir, '*.bed.gz'))
        if len(narrowpeak_files) < len(narrowpeak_files_csv):
            narrowpeak_files = narrowpeak_files_csv
            print("Reading from bed.gz files")
        else:
            print("Reading from parquet files")
        # read them in in parallel
        with mp.Pool(self.n_cores) as pool:
            results = pool.starmap(
                read_narrowpeak_file,
                tqdm([(fn, self.sample_name_part, self.split_on) for fn in narrowpeak_files], desc="Reading narrowPeak files")
                )
        samples = {sample_name: df for sample_name, df in results}
        return samples

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

        Returns:
        --------
        data_matrix : pandas.DataFrame
            DataFrame containing the aggregated data.
        """
        # Create BedTool object for regions
        regions_bed = pybedtools.BedTool.from_dataframe(self.all_regions_df)
        region_ids = self.all_regions_df['region_id'].unique()
        
        # Get samples
        samples = list(self.hmark_dict.keys())
        # Create an empty dataframe to store the results
        self.data_matrix = pd.DataFrame(index=samples, columns=region_ids)
        # Process each sample in parallel, using all available cores or the number of cores specified
        with mp.Pool(self.n_cores) as pool:
            results = pool.starmap(
                process_sample,  # fxn
                tqdm([(sample_item, regions_bed) for sample_item in self.hmark_dict.items()], desc="Creating data matrix"),  # args
                )
        # populate the data matrix
        for sample_name, mean_signal in tqdm(
            results, desc="Populating data matrix"
            ):
            if mean_signal is not None:
                self.data_matrix.loc[sample_name, mean_signal.index] = mean_signal
            else:
                self.data_matrix.loc[sample_name, :] = 0

        # convert all columns to float
        self.data_matrix = self.data_matrix.astype(float)
        # fill nans with 0
        self.data_matrix = self.data_matrix.fillna(0)
        # convert all columns to float
        self.data_matrix_w_meta = self.data_matrix.merge(
            self.metadata_df, left_index=True, right_index=True, how = 'left'
            ).dropna(subset = ['age_years', 'donor'])
        if self.overwrite_cached_data_matrix:
            # cache the data matrix
            os.makedirs(os.path.join(self.narrowpeak_dir, 'data_matrix'), exist_ok=True)
            self.data_matrix_w_meta.to_parquet(
                os.path.join(self.narrowpeak_dir, 'data_matrix', f"{self.name}_data_matrix_w_meta.parquet")
                )
        
    def _standardize_features(self):
        """
        Standardize the features using the mean and standard deviation of each column.
        """
        non_meta_cols = [col for col in self.data_matrix_w_meta.columns if col not in self.meta_cols]
        self.data_matrix_w_meta[non_meta_cols] = self.data_matrix_w_meta[non_meta_cols].apply(
            lambda x: (x - x.mean()) / x.std()
            )
        
    def predict(
        self, 
        target_column: str, 
        model, 
        result_column: str, 
        n_folds: int = 5,
        features: list = None,
        seperate_by_hmark: bool = False
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
        """
        # Prepare data
        y = self.data_matrix_w_meta[target_column]
        X = self.data_matrix_w_meta        

        # subset the data matrix to the specified features
        if features is not None:
            X = X[features + self.meta_cols]

        # split the data matrix into a dictionary of data matrices, keyed by histone mark
        if seperate_by_hmark:
            hmark_dict = {
                hmark: X.loc[X['histone_mark'] == hmark]
                for hmark in X['histone_mark'].unique()
                }
        else:
            hmark_dict = {'all': X}

        # drop the meta columns, except for 'donor'
        # so that we can keep samples from the same donors in the same fold
        meta_cols_to_drop = [col for col in self.meta_cols if col != 'donor']

        # train the model for each histone mark
        all_hmark_predictions = {}
        for hmark, hmark_df in hmark_dict.items():
            print(f"Training model for {hmark}", flush=True)
            h_mark_predictions = self._do_kfold_cv(
                model = model, 
                X = hmark_df.drop(columns=meta_cols_to_drop), 
                y = hmark_df[target_column], 
                n_folds = n_folds, 
                result_column = result_column + '_' + hmark
                )
            all_hmark_predictions.update(h_mark_predictions)
        
        # update the data matrix with the predictions
        self.data_matrix_w_meta[result_column] = self.data_matrix_w_meta.index.map(all_hmark_predictions)
            
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
        # check if we can split or don't have enough samples
        try:
            splitter = sgf.split(X, classes, groups=groups)
        except ValueError:
            print(f"Could not split data for {result_column}. Skipping.")
            self.trained_models[result_column].append(np.nan)
            y_pred_dict = {idx: np.nan for idx in X.index}
            return y_pred_dict
        # Perform cross-validation
        for _, (train_index, test_index) in enumerate(splitter):
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
        jobname = 'hstclck'
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
    seperate_by_hmark=True
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
        
        
    def plot_predictions_scatter(self, pred_col: str):
        # create a facet plot of regplots of each histone mark
        # don't share x or y axis
        # set font size to 12
        plt.rcParams.update({'font.size': 18})

        q = sns.lmplot(
            x='age_years', y='Histone age', 
            data=self.data_matrix_w_meta.loc[
                self.data_matrix_w_meta['sample_type'] == 'tissue'
                ].rename(columns = {pred_col: 'Histone age'}),
            hue = 'histone_mark', col = 'histone_mark', col_wrap = 4,
            scatter = False
            )

        # add pearson r to each facet
        for histone_mark, ax in q.axes_dict.items():
            subset = self.data_matrix_w_meta[
                (self.data_matrix_w_meta['histone_mark'] == histone_mark) 
                & ( self.data_matrix_w_meta['sample_type'] == 'tissue')
                ]
            r, p = pearsonr(subset['age_years'], subset[pred_col])
            rho, pval = spearmanr(subset['age_years'], subset[pred_col])
            ax.annotate(f"r = {r:.2f}, p = {p:.2e}", xy=(0.5, 0.1), xycoords=ax.transAxes, fontsize=12, color='black')
            # use grek letters for rho and pval
            ax.annotate(f"ρ = {rho:.2f}, p = {pval:.2e}", xy=(0.5, 0.15), xycoords=ax.transAxes, fontsize=12, color='black')
            
            # plot a pointplot of the mean histone age for each donor for this histone mark
            sns.pointplot(
                data = subset.rename(columns = {pred_col: 'Histone age'}),
                x = 'age_years', y = 'Histone age', linestyle='none', native_scale = True, color = 'black',
                legend = False, ax = ax
                )
