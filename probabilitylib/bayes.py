import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC, K2
from pgmpy.estimators import BayesianEstimator
from collections import defaultdict


def build_bayesian_network(pspace, scoring_method='BIC', n_samples=None, scale_factor=1,
                           random_state=42, verbose=False, bin_boundaries=None):
    from pgmpy.estimators import HillClimbSearch, BIC, K2, BayesianEstimator
    import pandas as pd
    import numpy as np

    # Get dataframe and drop NaNs
    df = pspace.df.dropna().copy()

    # --- Exclude weights from structural learning ---
    has_weights = "weights" in df.columns
    if has_weights:
        weights = df["weights"].copy()   # keep aside
        df = df.drop(columns=["weights"])

    # --- Identify variable types ---
    binary_vars = [col for col in df.columns if df[col].nunique() == 2]
    continuous_vars = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in binary_vars
    ]

    if verbose:
        print("Binary variables (treated as discrete):", binary_vars)
        print("Continuous variables (require discretization):", continuous_vars)

    # --- Handle continuous variables ---
    if len(continuous_vars) > 0:
        if bin_boundaries is None:
            raise ValueError(
                f"Continuous variables found: {continuous_vars}. "
                f"Please provide 'bin_boundaries' for discretization."
            )
        else:
            for col in continuous_vars:
                if col not in bin_boundaries:
                    raise ValueError(
                        f"Missing bin boundaries for continuous variable '{col}'. "
                        f"Expected keys for all continuous variables: {continuous_vars}"
                    )
                df[col] = pd.cut(
                    df[col],
                    bins=bin_boundaries[col],
                    labels=False,
                    include_lowest=True
                )

    def estimate_model(df, scoring_method):
        if scoring_method == 'BIC':
            score = BIC(df)
        else:
            raise ValueError("Unsupported scoring method. Choose 'BIC'.")

        hc = HillClimbSearch(df)
        model = hc.estimate(scoring_method=score)
        model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)
        return model, score

    def compute_edge_scores(model, df, score):
        edge_scores = {}
        if verbose:
            print("🔍 Model edges:", list(model.edges()))
        for edge in model.edges():
            child, parent = edge[1], edge[0]
            try:
                if child in df.columns and parent in df.columns:
                    edge_scores[edge] = score.local_score(child, [parent])
                else:
                    if verbose:
                        print(f"⚠️ Skipping edge {edge}: columns not in sampled df.")
            except Exception as e:
                if verbose:
                    print(f"❌ Error scoring edge {edge}: {e}")
        return edge_scores

    if n_samples is None:
        n_samples = int(scale_factor * len(weights))

    sampled_df = df.sample(
        n=n_samples,
        replace=True,
        weights=weights,
        random_state=random_state
    )

    if verbose:
        print("✅ Sampled df shape:", sampled_df.shape)
        print("✅ Sampled df columns:", sampled_df.columns.tolist())

    model, score = estimate_model(sampled_df, scoring_method)
    edge_scores = compute_edge_scores(model, sampled_df, score)
    return model, edge_scores

def build_bayesian_network_edges(chunk, *args, **kwargs):
    bin_boundaries = kwargs.get("bin_boundaries", None)
    sample_size = kwargs.get("sample_size", None)
    handle_missing = kwargs.get("handle_missing", 'drop')
    return_model = kwargs.get("return_model", False)
    pass
