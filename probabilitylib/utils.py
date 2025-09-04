import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random


def stream_dataframe(filepath, chunk_size, use_cols=None):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"🚫 The file at '{filepath}' was not found. Please check the path and try again.")
    
    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"🚫 The file '{filepath}' is not a CSV file. Only .csv files are supported.")

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=use_cols):
        yield chunk

def binning_from_chunks(filepath, chunk_size, continuous_columns=None, num_bins=4, sample_size=10000, use_cols=None):
    sample = []
    total_seen = 0

    for i, chunk in enumerate(tqdm(stream_dataframe(filepath, chunk_size, use_cols=use_cols), desc="Sampling chunks")):
        # Auto-detect continuous columns if not provided
        if i == 0 and continuous_columns is None:
            continuous_columns = chunk.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not continuous_columns:  # <-- skip if no numeric columns
            continue

        for _, row in chunk.iterrows():
            row_vals = row[continuous_columns].to_dict()
            total_seen += 1
            if len(sample) < sample_size:
                sample.append(row_vals)
            else:
                s = random.randint(0, total_seen - 1)
                if s < sample_size:
                    sample[s] = row_vals

    sample_df = pd.DataFrame(sample)

    bin_edges = {}
    for col in continuous_columns:
        quantiles = np.linspace(0, 1, num_bins + 1)
        edges = sample_df[col].dropna().quantile(quantiles).unique()
        edges.sort()  # Ensure order
        bin_edges[col] = edges

    return bin_edges



def bin_continuous_columns(pspace, columns, num_bins=4, strategy='quantile', return_edges=False):
    df_binned = pspace.df.copy()
    bin_edges = {}

    for col in columns:
        if strategy == 'quantile':
            try:
                edges = df_binned[col].dropna().quantile(np.linspace(0, 1, num_bins + 1)).unique()
                if len(edges) <= 1:
                    print(f"⚠️ Not enough unique values to bin column '{col}'")
                    continue
            except Exception as e:
                print(f"⚠️ Error binning column '{col}': {e}")
                continue
        elif strategy == 'uniform':
            min_val = df_binned[col].min()
            max_val = df_binned[col].max()
            edges = np.linspace(min_val, max_val, num_bins + 1)
        else:
            raise ValueError("Strategy must be 'quantile' or 'uniform'")

        edges = np.sort(edges)
        bin_edges[col] = edges
        df_binned[col] = pd.cut(df_binned[col], bins=edges, labels=False, include_lowest=True)

    if return_edges:
        return bin_edges
    else:
        return df_binned


def apply_bin_boundaries(pspace_or_df, bin_boundaries, handle_missing="error"):
    if hasattr(pspace_or_df, "df"):  
        df_copy = pspace_or_df.df.copy()
    else: 
        df_copy = pspace_or_df.copy()
    for col, bins in bin_boundaries.items():
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if handle_missing == "error" and df_copy[col].isna().any():
            raise ValueError(f"Missing values in column '{col}'.")
        elif handle_missing == "ignore":
            df_copy = df_copy.dropna(subset=[col])
        elif handle_missing == "fill":
            df_copy[col] = df_copy[col].fillna(-9999)
        df_copy[col] = pd.cut(df_copy[col], bins=bins, labels=False, include_lowest=True)

    return df_copy



