from collections import Counter, defaultdict
from functools import reduce
import numpy as np
import pandas as pd
import math
import scipy.stats
from scipy.stats import chi2_contingency
from pgmpy.models import DiscreteBayesianNetwork
from .bayes import build_bayesian_network
from .core import ProbabilitySpace
from .stats import IQR
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.base import DAG


def combine_means(results):
    total_weight = sum(chunk_weight for _, chunk_weight in results)
    weighted_sum = sum(mean * chunk_weight for mean, chunk_weight in results)
    return weighted_sum / total_weight if total_weight > 0 else None

def combine_variance(results):
    def combine_pair(r1, r2):
        m1, v1, w1 = r1
        m2, v2, w2 = r2
        delta = m2 - m1
        w_total = w1 + w2
        if w_total == 0:
            return 0.0, 0.0, 0.0
        new_mean = (w1 * m1 + w2 * m2) / w_total
        new_var = ((w1 * v1 + w2 * v2) + delta**2 * w1 * w2 / w_total) / w_total
        return new_mean, new_var, w_total

    _, combined_var, _ = reduce(combine_pair, results)
    return combined_var


def combine_IQR(results, variable):
    combined_df = pd.concat(results, ignore_index=True)
    pspace = ProbabilitySpace(combined_df)
    return IQR(pspace, variable)


def combine_conditional(results):
    total_numerator = sum(num for num, _ in results)
    total_denominator = sum(denom for _, denom in results)
    if total_denominator == 0:
        return 0.0
    return total_numerator / total_denominator if total_denominator else float("nan")


def combine_joint_probability(results, con=None):
    if con == "dependent":
        return sum(results)
    
    elif con == "independent":
        total_pA = 0
        total_pB = 0
        for pA, pB in results:
            total_pA += pA
            total_pB += pB
        return total_pA * total_pB


def combine_PDF_or_PMF(results, var_type="discrete", plot=True, **kwargs):
    
    if var_type == "discrete":
        all_values = results[0][0].index
        for r in results[1:]:
            all_values = all_values.union(r[0].index)

        pmf_list = [r[0].reindex(all_values, fill_value=0) for r in results]
        weights = np.array([r[1] for r in results]) 

        pmf_array = np.array([pmf.values for pmf in pmf_list])  

        weighted_pmf = pmf_array * weights[:, None]  

        combined = weighted_pmf.sum(axis=0)

        combined /= combined.sum()

        combined = pd.Series(combined, index=all_values)

        if plot:
            combined.sort_index().plot(kind='bar', title="Combined PMF")
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()
            return

        return combined


    elif var_type == "continuous":
        grid = results[0][0]
        pdf_matrix = np.array([pdf_vals for _, pdf_vals, _ in results])
        weights = np.array([w for _, _, w in results])
        total_weight = weights.sum()
        weighted_pdf = (pdf_matrix.T @ weights) / total_weight

        if plot:
            plt.plot(grid, weighted_pdf)
            plt.title("Combined PDF")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.grid(True)
            plt.show()
            return

        return grid, weighted_pdf

    else:
        raise ValueError("var_type must be 'discrete' or 'continuous'")


def combine_EDF_or_CDF(chunk_results, var_type="discrete", plot=False):
    if var_type == "discrete":
        
        combined_x = np.unique(np.concatenate([x for x, _ in chunk_results]))
        combined_weights = np.zeros_like(combined_x, dtype=float)

        for x_chunk, cum_chunk in chunk_results:
            weights_chunk = np.empty_like(cum_chunk)
            weights_chunk[0] = cum_chunk[0]
            weights_chunk[1:] = cum_chunk[1:] - cum_chunk[:-1]
            indices = np.searchsorted(combined_x, x_chunk)
            combined_weights[indices] += weights_chunk


        combined_cdf = np.cumsum(combined_weights)
        combined_cdf /= combined_cdf[-1]

        if plot:
            plt.step(combined_x, combined_cdf, where="post")
            plt.title("Combined EDF (Discrete)")
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.grid(True)
            plt.show()

        return combined_x, combined_cdf


    elif var_type == "continuous":
        grid = chunk_results[0][0]
        combined_weights = np.zeros_like(grid, dtype=float)
        
        for _, cdf_vals in chunk_results:
            weights_chunk = np.empty_like(cdf_vals)
            weights_chunk[0] = cdf_vals[0]
            weights_chunk[1:] = cdf_vals[1:] - cdf_vals[:-1]
            
            combined_weights += weights_chunk
        
        combined_cdf = np.cumsum(combined_weights)
        combined_cdf /= combined_cdf[-1]
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(grid, combined_cdf)
            plt.title("Combined EDF (Continuous)")
            plt.xlabel("Value")
            plt.ylabel("CDF")
            plt.grid(True)
            plt.show()
        
        return grid, combined_cdf

    else:
        raise ValueError(f"Unsupported var_type: {var_type}")


def combine_independence(results):
    aggregated_table = None

    for result in results:
        if result is not None and not result.empty:
            if aggregated_table is None:
                aggregated_table = result.copy()
            else:
                aggregated_table = aggregated_table.add(result, fill_value=0)

    if aggregated_table is None:
        return None

    aggregated_table = aggregated_table.fillna(0)

    statistic, p_value, dof, _ = chi2_contingency(aggregated_table)

    return {
        "statistic": statistic,
        "p_value": round(p_value, 9),
        "dof": dof
    }

def combine_mutual_information_outputs(outputs):
    combined_joint = {}
    combined_marginal_x = {}
    combined_marginal_y = {}
    combined_total_weight = 0.0
    
    for joint, marginal_x, marginal_y, total_weight in outputs:
        combined_total_weight += total_weight
        
        for k, v in joint.items():
            combined_joint[k] = combined_joint.get(k, 0) + v
        
        for k, v in marginal_x.items():
            combined_marginal_x[k] = combined_marginal_x.get(k, 0) + v
        
        for k, v in marginal_y.items():
            combined_marginal_y[k] = combined_marginal_y.get(k, 0) + v
    
    return combined_joint, combined_marginal_x, combined_marginal_y, combined_total_weight


def combine_mutual_information(outputs):
    combined_joint, combined_marginal_x, combined_marginal_y, combined_total_weight = combine_mutual_information_outputs(outputs)

    mi = 0.0
    for (x, y), joint_count in combined_joint.items():
        p_xy = joint_count / combined_total_weight
        p_x = combined_marginal_x.get(x, 0) / combined_total_weight
        p_y = combined_marginal_y.get(y, 0) / combined_total_weight
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    
    return mi


def combine_bayesian_models(results, top_k=10, score_type='hybrid', return_all=False):
    edge_softmax_sum = defaultdict(float)
    edge_freq = defaultdict(int)

    for result in results:
        if not result or not isinstance(result, tuple) or len(result) != 2:
            continue
        
        model, bic_scores = result
        if bic_scores is None or len(bic_scores) == 0:
            continue

        neg_bics = np.array([-v for v in bic_scores.values()])
        softmax_scores = np.exp(neg_bics - np.max(neg_bics))
        softmax_scores /= softmax_scores.sum()

        for (edge, _), soft_score in zip(bic_scores.items(), softmax_scores):
            edge_softmax_sum[edge] += soft_score
            edge_freq[edge] += 1

    max_soft = max(edge_softmax_sum.values()) if edge_softmax_sum else 1.0
    max_freq = max(edge_freq.values()) if edge_freq else 1.0

    all_edges = set(edge_softmax_sum.keys()).union(edge_freq.keys())

    undirected_pairs = defaultdict(lambda: {'forward': None, 'reverse': None})
    for edge in all_edges:
        u, v = edge
        pair = tuple(sorted((u, v)))
        score_data = {
            'freq': edge_freq[edge],
            'soft': edge_softmax_sum[edge],
        }

        # Choose final score based on score_type
        if score_type == 'softmax':
            final_score = score_data['soft']
        elif score_type == 'frequency':
            final_score = score_data['freq']
        elif score_type == 'hybrid':
            norm_freq = score_data['freq'] / max_freq
            norm_soft = score_data['soft'] / max_soft
            final_score = (norm_freq + norm_soft) / 2
        else:
            raise ValueError("score_type must be 'softmax', 'frequency', or 'hybrid'")

        score_data['final'] = final_score

        if edge == pair:
            undirected_pairs[pair]['forward'] = (edge, score_data)
        else:
            undirected_pairs[pair]['reverse'] = (edge, score_data)

    final_scores = {}
    for pair, directions in undirected_pairs.items():
        forward = directions['forward']
        reverse = directions['reverse']

        if forward and reverse:
            if forward[1]['final'] >= reverse[1]['final']:
                final_scores[forward[0]] = forward[1]['final']
            else:
                final_scores[reverse[0]] = reverse[1]['final']
        elif forward:
            final_scores[forward[0]] = forward[1]['final']
        elif reverse:
            final_scores[reverse[0]] = reverse[1]['final']

    sorted_edges = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_edges = [edge for edge, _ in sorted_edges[:top_k]]

    final_model = nx.DiGraph()
    for u, v in top_edges:
        final_model.add_nodes_from([u, v])
        final_model.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(final_model):
            final_model.remove_edge(u, v)

    final_model_edges = list(final_model.edges())

    if return_all:
        return final_model_edges, sorted_edges
    return final_model_edges



def aggregate_and_build_network(
    sampled_chunks,
    scoring_method='BIC',
    n_samples=None,
    scale_factor=1,
    random_state=42,
    n_iterations=100,
    consensus_threshold=0.6,
    bin_boundaries=None 
):

    combined_df = pd.concat(sampled_chunks, ignore_index=True)
    pspace = ProbabilitySpace(combined_df)
    print(f"Combined sampled dataset shape: {combined_df.shape}")

    edge_counts = Counter()

    for i in range(n_iterations):
        run_random_state = None if random_state is None else random_state + i

        model, _ = build_bayesian_network(
            pspace,
            scoring_method=scoring_method,
            n_samples=n_samples,
            scale_factor=scale_factor,
            random_state=run_random_state,
            bin_boundaries=bin_boundaries   
        )

        for edge in model.edges():
            edge_counts[edge] += 1

    required_count = int(n_iterations * consensus_threshold)
    consensus_edges = [edge for edge, count in edge_counts.items() if count >= required_count]

    print(f"\n📊 Consensus edges (≥{int(consensus_threshold * 100)}% of runs): {consensus_edges}")

    return consensus_edges, edge_counts
