import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
import scipy.stats
import numpy as np
from scipy.stats import chi2_contingency
from collections import defaultdict
import math


def independence(pspace, col1, col2, bin_boundaries=None, scale=1000, crosstab=False, discrete_threshold=10):
    df = pspace.df[[col1, col2, "weights"]].dropna()

    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Error: Continuous variable '{col}' with {num_unique} unique values "
                    "requires bin boundaries. Please provide bin_boundaries to discretize it."
                )

    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = np.sort(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    for col in [col1, col2]:
        df[col] = pd.Categorical(df[col]).codes

    table = pd.pivot_table(
        df,
        values="weights",
        index=col1,
        columns=col2,
        aggfunc="sum",
        fill_value=0
    )

    table *= scale

    if crosstab:
        return table
    else:
        statistic, p_value, dof, _ = chi2_contingency(table)
        return {"statistic": statistic, "p_value": p_value, "dof": dof}



def mutual_information(pspace, col1, col2, bin_boundaries=None, discrete_threshold=10):
    df = pspace.df[[col1, col2, "weights"]].dropna().copy()
    total_weight = df["weights"].sum()

    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Error: Continuous variable '{col}' with {num_unique} unique values "
                    "requires bin boundaries. Please provide bin_boundaries to discretize it."
                )

    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = np.sort(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    for col in [col1, col2]:
        df[col] = pd.Categorical(df[col]).codes

    joint = df.groupby([col1, col2])["weights"].sum().reset_index().rename(columns={"weights": "p_x_y"})
    joint["p_x_y"] = joint["p_x_y"] / total_weight

    p_x = df.groupby([col1])["weights"].sum().reset_index().rename(columns={"weights": "p_x"})
    p_x["p_x"] = p_x["p_x"] / total_weight

    p_y = df.groupby([col2])["weights"].sum().reset_index().rename(columns={"weights": "p_y"})
    p_y["p_y"] = p_y["p_y"] / total_weight

    joint = joint.merge(p_x, on=col1).merge(p_y, on=col2)

    joint["contrib"] = joint["p_x_y"] * np.log2(joint["p_x_y"] / (joint["p_x"] * joint["p_y"]))

    return joint["contrib"].sum()


def accumulate_joint_and_marginals(pspace, col1, col2, bin_boundaries=None, discrete_threshold=10):
    df = pspace.df[[col1, col2, "weights"]].dropna().copy()

    binned_columns = set(bin_boundaries.keys()) if bin_boundaries else set()

    for col in [col1, col2]:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in binned_columns:
            num_unique = df[col].nunique()
            if num_unique > discrete_threshold:
                raise ValueError(
                    f"Continuous variable '{col}' with {num_unique} unique values requires bin boundaries."
                )

    if bin_boundaries:
        for col in [col1, col2]:
            if col in bin_boundaries:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                bins = sorted(bin_boundaries[col])
                df[col] = pd.cut(df[col], bins=bins, include_lowest=True, right=False).astype(str)

    for col in [col1, col2]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str)

    joint_counts = df.groupby([col1, col2])['weights'].sum().to_dict()
    marginal_x = df.groupby(col1)['weights'].sum().to_dict()
    marginal_y = df.groupby(col2)['weights'].sum().to_dict()
    total_weight = df['weights'].sum()

    return joint_counts, marginal_x, marginal_y, total_weight
