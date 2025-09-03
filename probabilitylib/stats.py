import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
import scipy.stats
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import warnings

def expected_mean(pspace, col):
    df = pspace.df
    mask = df[col].notna()
    weighted_sum = (df.loc[mask, col] * df.loc[mask, 'weights']).sum()
    total_weight = df.loc[mask, 'weights'].sum()
    return weighted_sum / total_weight if total_weight != 0 else np.nan


def variance(pspace, col):
    df = pspace.df
    mask = df[col].notna()
    mean = expected_mean(pspace, col)
    weighted_sq = (df.loc[mask, col] ** 2 * df.loc[mask, 'weights']).sum()
    total_weight = df.loc[mask, 'weights'].sum()
    return weighted_sq / total_weight - mean ** 2 if total_weight != 0 else np.nan


def IQR(pspace, variable):
    """
    Compute the weighted interquartile range (IQR) for a given variable in a probability space.

    Parameters:
        pspace: ProbabilitySpace object with attributes:
            - pspace.df: DataFrame containing the data
            - 'weights': a column in the DataFrame representing the weight/probability of each row
        variable: string, name of the variable to compute the IQR for

    Returns:
        dict with Q25, Q75, and IQR values, or np.nan if total weight is zero.
    """
    df = pspace.df
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame.")

    values = np.asarray(df[variable].values)
    weights = np.asarray(df["weights"].values)

    # Remove rows with NaN values
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]

    # Sort values and weights
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative distribution
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    if total_weight == 0:
        return np.nan

    cdf = cumulative_weights / total_weight

    # Compute weighted quantiles
    q25 = np.interp(0.25, cdf, sorted_values)
    q75 = np.interp(0.75, cdf, sorted_values)

    return {"Q25": q25, "Q75": q75, "IQR": q75 - q25}


def conditional(pspace, event_condition, given_condition=None):
    df = pspace.df

    # evaluate if callable, else assume Series
    event_mask = event_condition(df) if callable(event_condition) else event_condition
    given_mask = given_condition(df) if callable(given_condition) else given_condition

    # align to df.index
    event_mask = event_mask.reindex(df.index, fill_value=False)
    if given_mask is None:
        given_mask = pd.Series(True, index=df.index)
    else:
        given_mask = given_mask.reindex(df.index, fill_value=False)

    numerator = (df["weights"][event_mask & given_mask]).sum()
    denominator = (df["weights"][given_mask]).sum()

    if denominator == 0:
        return 0.0

    return numerator / denominator


def joint_probability(pspace, condition_A, condition_B, con="dependent"):
    """
    Calculates joint or independent probability in a ProbabilitySpace.

    Parameters
    ----------
    pspace : ProbabilitySpace
        The probability space object containing a dataframe `df` with a 'weights' column.
    condition_A : pd.Series of bool
        Boolean mask for event A.
    condition_B : pd.Series of bool
        Boolean mask for event B.
    con : str, default "dependent"
        "dependent" -> compute P(A and B)
        "independent" -> compute P(A) * P(B)

    Returns
    -------
    float
        Joint or independent probability.
    """
    if con not in {"dependent", "independent"}:
        raise ValueError("`con` must be 'dependent' or 'independent'")

    if condition_A is None or condition_B is None:
        raise ValueError("Error: conditional statements must be included for both variables")

    df = pspace.df

    if con == "dependent":
        # Joint probability P(A and B)
        return df["weights"][(condition_A) & (condition_B)].sum()
    
    elif con == "independent":
        # Multiply marginal probabilities P(A)*P(B)
        pA = df["weights"][condition_A].sum()
        pB = df["weights"][condition_B].sum()
        return pA * pB


def PDF_or_PMF(pspace, variable, var_type="discrete", plot=True, bandwidth=None):
    df = pspace.df[[variable, 'weights']].dropna(subset=[variable])
    x = df[variable].values
    weights = df['weights'].values

    if var_type == "discrete":
        # PMF as weighted sums grouped by values
        pmf = pd.Series(weights, index=x).groupby(level=0).sum()
        pmf /= pmf.sum()

        if plot:
            pmf.sort_index().plot(kind='bar', title=f"PMF of {variable}")
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()
            return
        else:
            return pmf

    elif var_type == "continuous":
        var = variance(pspace, variable)
        sd = var**0.5
        iqr = IQR(pspace, variable)
        n_samples = pspace.size()
        scale = min(sd, iqr["IQR"] / 1.34)
        
        if bandwidth is not None:
            kde = gaussian_kde(x, weights=weights, bw_method=bandwidth)

        else:
            silverman_bw = 1.06 * scale * n_samples ** (-1 / 5)
    
            kde_default = gaussian_kde(x, weights=weights)
            default_bw = kde_default.factor * x.std(ddof=1)
    
            bw_scaling_factor = silverman_bw / default_bw
            
            kde = gaussian_kde(x, weights=weights, bw_method=bw_scaling_factor)
        
        x_vals = np.linspace(x.min(), x.max(), 500)
        pdf_vals = kde(x_vals)

        if plot:
            plt.plot(x_vals, pdf_vals)
            plt.title(f"PDF of {variable}")
            plt.xlabel(variable)
            plt.ylabel("Density")
            plt.grid(True)
            plt.show()
            return
        else:
            return x_vals, pdf_vals
    else:
        raise ValueError("var_type must be 'discrete' or 'continuous'")


def EDF_or_CDF(pspace, variable, var_type="discrete", plot=True, grid_points=500):    
    df = pspace.df[[variable, 'weights']].dropna(subset=[variable])
    
    if pd.api.types.is_categorical_dtype(df[variable]) or df[variable].dtype == "object":
        warnings.warn(
            f"Variable '{variable}' appears to be categorical. Please convert it to a numerical or ordinal format."
        )

    x = df[variable].values
    weights = df['weights'].values
    total_weight = np.sum(weights)

    if var_type == "discrete":
        grouped_df = df.groupby(variable)['weights'].sum().reset_index()
        grouped_df = grouped_df.sort_values(by=variable)

        x_sorted = grouped_df[variable].values
        y_cdf = np.cumsum(grouped_df['weights'].values)

        y_cdf = y_cdf / total_weight

        if plot:
            plt.step(x_sorted, y_cdf, where='post')
            plt.title(f"Empirical CDF of {variable} (discrete)")
            plt.xlabel(variable)
            plt.ylabel("Cumulative Probability")
            plt.grid(True)
            plt.show()
            return
        else:
            return x_sorted, y_cdf

    elif var_type == "continuous":
        x_min = x.min()
        x_max = x.max()
        x_vals = np.linspace(x_min, x_max, grid_points)

        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        weights_sorted = weights[sorted_indices]

        cumsum_weights = np.cumsum(weights_sorted)

        indices = np.searchsorted(x_sorted, x_vals, side='right') - 1
        cdf_vals = np.zeros_like(x_vals)

        valid = indices >= 0
        cdf_vals[valid] = cumsum_weights[indices[valid]]

        # Normalize
        cdf_vals = cdf_vals / total_weight

        if plot:
            plt.plot(x_vals, cdf_vals)
            plt.title(f"Empirical CDF of {variable} (continuous)")
            plt.xlabel(variable)
            plt.ylabel("Cumulative Probability")
            plt.grid(True)
            plt.show()
            return
        else:
            return x_vals, cdf_vals

    else:
        raise ValueError("var_type must be 'discrete' or 'continuous'")



