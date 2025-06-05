import pandas as pd
import argparse

mm10_nc_to_chr = {
    'NC_000067.6': 'chr1',
    'NC_000068.7': 'chr2',
    'NC_000069.6': 'chr3',
    'NC_000070.6': 'chr4',
    'NC_000071.6': 'chr5',
    'NC_000072.6': 'chr6',
    'NC_000073.6': 'chr7',
    'NC_000074.6': 'chr8',
    'NC_000075.6': 'chr9',
    'NC_000076.6': 'chr10',
    'NC_000077.6': 'chr11',
    'NC_000078.6': 'chr12',
    'NC_000079.6': 'chr13',
    'NC_000080.6': 'chr14',
    'NC_000081.6': 'chr15',
    'NC_000082.6': 'chr16',
    'NC_000083.6': 'chr17',
    'NC_000084.6': 'chr18',
    'NC_000085.6': 'chr19',
    'NC_000086.7': 'chrX',
    'NC_000087.7': 'chrY',
}

def read_methylation_file(file_path):
    """
    Read methylation data file into a pandas DataFrame and map NCBI IDs to chromosomes.
    
    Parameters:
    -----------
    file_path : str
        Path to the methylation data file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: chromosome, position, methylation_percentage, coverage
    """
    
    # Read the file with tab separator
    df = pd.read_csv(file_path, sep='\s+')
    
    # Split the index column into stuff and position
    df[['stuff', 'start']] = df['index'].str.split(':', expand=True)
    
    # Rename second and third columns for clarity
    df = df.rename(columns={
        df.columns[1]: 'mf',
        df.columns[2]: 'cov'
    })
    
    # Extract NCBI ID from stuff column
    df['ncbi_id'] = df['stuff'].str.split('|').str[3]
    
    # Map NCBI IDs to chromosomes using the dictionary
    df['chromosome'] = df['ncbi_id'].map(mm10_nc_to_chr)
    
    # Convert position to integer type
    df['start'] = df['start'].astype(int)
    
    # Reorder and select final columns
    df['end'] = df['start'] + 1
    df['methyl_reads'] = (df['mf']/100) * df['cov']
    df['unmethyl_reads'] = (1 - (df['mf']/100)) * df['cov']
    df = df[['chromosome', 'start', 'end', 'mf',  'methyl_reads', 'unmethyl_reads']]
    
    # Drop rows where chromosome is None
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process RRBS methylation file and map NCBI IDs to chromosomes.')
    parser.add_argument('file_path', help='Path to the methylation data file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    df = read_methylation_file(args.file_path)
    
    # write to parquet, not including index or header
    df.to_parquet(args.file_path.replace('.txt.gz', '.parquet'), index=False)
    print(f"Wrote {args.file_path.replace('.txt.gz', '.parquet')}")