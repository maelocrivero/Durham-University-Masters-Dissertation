from .core import ProbabilitySpace
from .utils import stream_dataframe
from .stats import expected_mean, EDF_or_CDF, joint_probability
from .bayes import build_bayesian_network
from .dependency import accumulate_joint_and_marginals
from tqdm import tqdm
import pandas as pd
from collections import Counter
import numpy as np
import random
import probabilitylib as pl
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz



def get_total_weights(filepath, chunk_size=10000, use_cols=None):
    total_weights = 0
    for chunk in stream_dataframe(filepath, chunk_size=chunk_size, use_cols=use_cols):
        if "weights" in chunk.columns:
            total_weights += chunk["weights"].sum()
        else:
            total_weights += len(chunk)
    return total_weights


def process_in_chunks(filepath, chunk_size, func, *args, random_state=42, use_cols=None, **kwargs):
    if use_cols is not None:
        use_cols = set(use_cols)
        use_cols.add("weights")
        use_cols = list(use_cols)

    total_weights = get_total_weights(filepath, use_cols=use_cols)

    results = []
    for i, chunk in enumerate(tqdm(stream_dataframe(filepath, chunk_size, use_cols=use_cols), desc="Processing chunks")):
        chunk = chunk.copy()
    
        if "weights" not in chunk.columns:
            chunk["weights"] = np.ones(len(chunk))
        
        chunk = chunk[chunk["weights"].notna() & (chunk["weights"] > 0)].copy()
    
        chunk["weights"] = chunk["weights"] / total_weights
    
        try:
            result = _apply_func(chunk, func, *args, **kwargs)
            results.append(result)
            print(f"Chunk weights sum: {chunk['weights'].sum():.6f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Error in chunk {i+1}: {e}")


    return results



def _apply_func(chunk, func, *args, **kwargs):
    if func.__name__ == "expected_mean":
        ps = ProbabilitySpace(chunk, normalise=False)
        val = func(ps, *args, **kwargs)
        return (val, chunk["weights"].sum())

    elif func.__name__ == "variance":
        ps = ProbabilitySpace(chunk, normalise=False)
        mean = expected_mean(ps, args[0])
        var = func(ps, *args, **kwargs)
        return (mean, var, chunk["weights"].sum())

    elif func.__name__ == "mutual_information":
        ps = ProbabilitySpace(chunk, normalise=False)
        return accumulate_joint_and_marginals(ps, *args, **kwargs)

    elif func.__name__ == "IQR":
        sample_size = kwargs.get("sample_size", None)
        var = args[0]
        df = chunk[[var, "weights"]].dropna()
    
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=kwargs.get("random_state", 42))
    
        return df

    elif func.__name__ == "joint_probability":
        con = kwargs.get("con", "dependent")
        ps = ProbabilitySpace(chunk, normalise=False)
    
        condition_A = args[0](chunk) if callable(args[0]) else args[0]
        condition_B = args[1](chunk) if callable(args[1]) else args[1]
    
        if con == "dependent":
            return ps.df["weights"][(condition_A) & (condition_B)].sum()
        
        elif con == "independent":
            pA = ps.df["weights"][condition_A].sum()
            pB = ps.df["weights"][condition_B].sum()
            return pA, pB



    elif func.__name__ == "conditional":
        ps = ProbabilitySpace(chunk, normalise=False)
        df = ps.df
        
        event_mask = args[0](df)      
        given_mask = args[1](df) if args[1] is not None else pd.Series(True, index=df.index)
        
        numerator = (df["weights"][event_mask & given_mask]).sum()
        denominator = (df["weights"][given_mask]).sum()
        
        if denominator == 0:
            return 0.0
        return numerator, denominator


    elif func.__name__ == "EDF_or_CDF":
        variable = args[0]
        var_type = kwargs.get("var_type", None)
    
        chunk = chunk.dropna(subset=[variable])
    
        if var_type == "discrete":
            ps = ProbabilitySpace(chunk, normalise=False)
            grouped_df = ps.df.groupby(variable)['weights'].sum().reset_index()
            grouped_df = grouped_df.sort_values(by=variable)
        
            x_values = grouped_df[variable].values
            weights_chunk = grouped_df['weights'].values
            cum_weights = np.cumsum(weights_chunk)
        
            return x_values, cum_weights

    
        elif var_type == "continuous":
            global_min = kwargs.get("global_min")
            global_max = kwargs.get("global_max")
            num_points = kwargs.get("num_points", 500)
    
            if global_min is None or global_max is None:
                raise ValueError("global_min and global_max must be provided")
    
            x = chunk[variable].values
            weights = chunk['weights'].values
    
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            weights_sorted = weights[sorted_indices]
    
            cumsum_weights = np.cumsum(weights_sorted)
    
            grid = np.linspace(global_min, global_max, num_points)
            indices = np.searchsorted(x_sorted, grid, side='right') - 1
            cdf_vals = np.zeros_like(grid)
            valid = indices >= 0
            cdf_vals[valid] = cumsum_weights[indices[valid]]
    
            total_weight = weights.sum()
    
            return grid, cdf_vals
           
    elif func.__name__ == "PDF_or_PMF":
        variable = args[0]
        var_type = kwargs.get("var_type", None)
        ps = ProbabilitySpace(chunk, normalise=False)
    
        if var_type is None:
            raise ValueError("Must specify var_type as 'discrete' or 'continuous'")
    
        if var_type == "discrete":
            df = ps.df[[variable, 'weights']].dropna(subset=[variable])
            x = df[variable].values
            weights = df['weights'].values
            pmf = pd.Series(weights, index=x).groupby(level=0).sum()
            return pmf, pmf.sum()
    
        elif var_type == "continuous":
            global_min = kwargs.get("global_min")
            global_max = kwargs.get("global_max")
            sd = kwargs.get("sd")
            iqr = kwargs.get("iqr")
            n_samples = kwargs.get("n_samples")
            grid_size = kwargs.get("grid_size", 1000)
            bandwidth = kwargs.get("bandwidth", None)
        
            if None in (sd, iqr, n_samples, global_min, global_max):
                raise ValueError("Must provide sd, iqr, n_samples, global_min, global_max")
        
            df = ps.df[[variable, 'weights']].dropna(subset=[variable])
            x = df[variable].values
            weights = df['weights'].values

            if bandwidth is not None:
                kde = gaussian_kde(x, weights=weights, bw_method=bw_scaling_factor)

            else:        
                scale = min(sd, iqr / 1.34)
                silverman_bw = 1.06 * scale * n_samples ** (-1 / 5)
            
                kde_default = gaussian_kde(x, weights=weights)
                default_bw = kde_default.factor * x.std(ddof=1)
                bw_scaling_factor = silverman_bw / default_bw
            
                kde = gaussian_kde(x, weights=weights, bw_method=bw_scaling_factor)
        
            x_grid = np.linspace(global_min, global_max, grid_size)
            pdf_vals = kde.evaluate(x_grid)
            total_weight = weights.sum()
        
            return x_grid, pdf_vals, total_weight
         
    elif func.__name__ == "independence":
        ps = ProbabilitySpace(chunk, normalise=False)
        result = func(ps, *args, **kwargs, crosstab=True)
        return result

    elif func.__name__ == "build_bayesian_network":
        sample_size = kwargs.get("sample_size", None)
        handle_missing = kwargs.get("handle_missing", 'drop')
        
        if handle_missing == 'drop':
            chunk = chunk.dropna()
        elif handle_missing == 'fill':
            chunk = chunk.fillna(0)  
    
        if sample_size is not None and len(chunk) > sample_size:
            chunk = chunk.sample(
                n=sample_size,
                weights=chunk['weights'],
                random_state=kwargs.get("random_state", 42)
            )
    
        return chunk
        
    elif func.__name__ == "build_bayesian_network_edges":
        sample_size = kwargs.get("sample_size", None)
        scoring_method = kwargs.get("scoring_method", "BIC")
        bin_boundaries = kwargs.get("bin_boundaries", None)
    
        chunk_pspace = ProbabilitySpace(chunk, normalise=False)
    
        model, bic_scores = build_bayesian_network(
            chunk_pspace,
            n_samples=sample_size,
            scoring_method=scoring_method,
            bin_boundaries=bin_boundaries
        )
    
        return model, bic_scores

    else:
        return func(chunk, *args, **kwargs)
