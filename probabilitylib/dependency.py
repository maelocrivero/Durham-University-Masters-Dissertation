import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
import scipy.stats
import numpy as np
from scipy.stats import chi2_contingency
from collections import defaultdict
import math


def independence(pspace, col1, col2, bin_boundaries=None, scale=1000, crosstab=False, discrete_threshold=10):
    import pandas as pd
    import numpy as np
    from scipy.stats import chi2_contingency

    df = pspace.df[[col1, col2, "weights"]].dropna()

    # Determine which columns are binned
    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    # Check numeric variables without bin boundaries
    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Error: Continuous variable '{col}' with {num_unique} unique values "
                    "requires bin boundaries. Please provide bin_boundaries to discretize it."
                )
            # Otherwise, treat as discrete (small cardinality numeric)

    # Apply binning if bin_boundaries provided
    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = np.sort(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    # Convert categorical or discrete numeric variables to numeric codes
    for col in [col1, col2]:
        df[col] = pd.Categorical(df[col]).codes

    # Build weighted contingency table
    table = pd.pivot_table(
        df,
        values="weights",
        index=col1,
        columns=col2,
        aggfunc="sum",
        fill_value=0
    )

    # Scale weights
    table *= scale

    if crosstab:
        return table
    else:
        # Chi-squared test
        statistic, p_value, dof, _ = chi2_contingency(table)
        return {"statistic": statistic, "p_value": p_value, "dof": dof}



def mutual_information(pspace, col1, col2, bin_boundaries=None, discrete_threshold=10):
    import pandas as pd
    import numpy as np

    df = pspace.df[[col1, col2, "weights"]].dropna().copy()
    total_weight = df["weights"].sum()

    # Determine which columns are binned
    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    # Check numeric variables without bin boundaries
    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Error: Continuous variable '{col}' with {num_unique} unique values "
                    "requires bin boundaries. Please provide bin_boundaries to discretize it."
                )
            # Otherwise, treat as discrete (low cardinality numeric)

    # Apply binning if bin_boundaries provided
    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = np.sort(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    # Convert discrete/categorical variables to codes
    for col in [col1, col2]:
        df[col] = pd.Categorical(df[col]).codes

    # Compute weighted probabilities
    joint = df.groupby([col1, col2])["weights"].sum().reset_index().rename(columns={"weights": "p_x_y"})
    joint["p_x_y"] = joint["p_x_y"] / total_weight

    p_x = df.groupby([col1])["weights"].sum().reset_index().rename(columns={"weights": "p_x"})
    p_x["p_x"] = p_x["p_x"] / total_weight

    p_y = df.groupby([col2])["weights"].sum().reset_index().rename(columns={"weights": "p_y"})
    p_y["p_y"] = p_y["p_y"] / total_weight

    # Merge joint and marginal probabilities
    joint = joint.merge(p_x, on=col1).merge(p_y, on=col2)

    # Compute mutual information contributions
    joint["contrib"] = joint["p_x_y"] * np.log2(joint["p_x_y"] / (joint["p_x"] * joint["p_y"]))

    return joint["contrib"].sum()


def accumulate_joint_and_marginals(pspace, col1, col2, bin_boundaries=None, discrete_threshold=10):
    """
    Process a chunk of data and return weighted joint and marginal counts,
    with consistent binning/discrete-variable logic.

    Parameters:
        pspace: ProbabilitySpace containing the chunk of data.
        col1 (str): First variable.
        col2 (str): Second variable.
        bin_boundaries (dict, optional): Dictionary with bin edges for col1 and/or col2.
        discrete_threshold (int): Max unique values for a numeric variable to be treated as discrete.

    Returns:
        joint_counts (dict): Weighted joint counts of (col1, col2).
        marginal_x (dict): Weighted marginal counts for col1.
        marginal_y (dict): Weighted marginal counts for col2.
        total_weight (float): Total sum of weights in this chunk.
    """
    import pandas as pd

    df = pspace.df[[col1, col2, "weights"]].dropna().copy()

    # Determine which columns are binned
    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    # Check numeric variables without bin boundaries
    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Continuous variable '{col}' with {num_unique} unique values requires bin boundaries."
                )
            # otherwise, treat as discrete (small cardinality numeric)

    # Apply binning if bin_boundaries provided
    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = sorted(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    # Convert discrete or categorical variables to strings for grouping
    for col in [col1, col2]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Compute weighted joint and marginal counts
    joint_counts = df.groupby([col1, col2])['weights'].sum().to_dict()
    marginal_x = df.groupby(col1)['weights'].sum().to_dict()
    marginal_y = df.groupby(col2)['weights'].sum().to_dict()
    total_weight = df['weights'].sum()

    return joint_counts, marginal_x, marginal_y, total_weight
