#!/usr/bin/env python3

import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm

def combine_parquet_files(input_dir):
    """Combine all parquet files in a directory that don't contain 'old'
    
    Parameters
    ----------
    input_dir : str
        Directory containing parquet files to combine
    """
    # Get list of all parquet files, excluding those with 'old' in the name
    parquet_files = [f for f in glob.glob(os.path.join(input_dir, "*.parquet")) 
                    if 'old' not in os.path.basename(f)]
    
    output_file = os.path.join(input_dir, "data_matrix_w_meta.parquet")
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")
    # Initialize an empty list to store dataframes
    dfs = []

    # Read each parquet file
    print("Reading and combining parquet files...")
    for file in tqdm(parquet_files):
        try:
            df = pd.read_parquet(file, thrift_string_size_limit=1000000000)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            continue

    if not dfs:
        print("No data frames were successfully read")
        return
        
    # Combine all dataframes
    print("Concatenating dataframes...")
    combined_df = pd.concat(dfs, axis=0)

    # Write the combined dataframe to a new parquet file
    print(f"Writing combined data to {output_file}...")
    combined_df.to_parquet(output_file)

    print("Done! Final shape:", combined_df.shape)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Combine parquet files in a directory')
    parser.add_argument('input_dir', type=str, help='Directory containing parquet files to combine')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist")
        return
    
    combine_parquet_files(args.input_dir)

if __name__ == "__main__":
    main()