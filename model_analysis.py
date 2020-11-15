import math
import statistics

import numpy as np
from scipy import stats as stats


def compare_two_models(model_idx1, model_idx2, df):
    alfa = 0.05
    df = df.sort_values('mean_accuracy', ascending=False)
    row1 = df.iloc[model_idx1]
    row2 = df.iloc[model_idx2]
    t, p_value = paired_5x2_ttest(row1, row2)
    print(
        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]}, accuracy={row1["mean_accuracy"]}] '
        f'with [k={row2["n_neighbors"]}, m={row2["metric"]}, f={row2["n_features"]}, accuracy={row2["mean_accuracy"]}]')
    print(f'\tt-statistic: {t}, p-value: {p_value}, alfa: {alfa}')

    if p_value < alfa:
        print('\tIt is statistically significant!')
    else:
        print('\tIt is NOT statistically significant!')


def find_best_statistically_significant_model(df):
    alfa = 0.05
    df = df.sort_values('mean_accuracy', ascending=False)
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and i < j:
                t, p_value = paired_5x2_ttest(row1, row2)
                if p_value < alfa:
                    print(
                        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]},'
                        f' accuracy={row1["mean_accuracy"]}] with [k={row2["n_neighbors"]}, m={row2["metric"]},'
                        f' f={row2["n_features"]}, accuracy={row2["mean_accuracy"]}]')
                    print(f'\tt-statistic: {t}, p-value: {p_value}, alfa: {alfa}')
                    return


def compare_every_model_paired(df):
    alfa = 0.05
    statistical_significant_pairs = 0
    statistical_insignificant_pairs = 0
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and i < j:
                t, p_value = paired_5x2_ttest(row1, row2)
                if p_value < alfa:
                    statistical_significant_pairs = statistical_significant_pairs + 1
                    print(
                        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]}] with'
                        f' [k={row2["n_neighbors"]}, m={row2["metric"]}, f={row2["n_features"]}]')
                    print(f'\tt-statistic: {t}, p-value: {p_value}, alfa: {alfa} -> {p_value} (p) < {alfa} (a)')
                else:
                    statistical_insignificant_pairs = statistical_insignificant_pairs + 1

    print(
        f'Statistical significant pairs = {statistical_significant_pairs}, '
        f'statistical insignificant pairs = {statistical_insignificant_pairs}')


def paired_5x2_ttest(m1, m2):
    p_1_1 = m1['scores'][0] - m2['scores'][0]
    variances = []
    for i in range(0, 5 * 2, 2):  # 5 repeats of 2-fold CV
        p1 = m1['scores'][i] - m2['scores'][i]
        p2 = m1['scores'][i + 1] - m2['scores'][i + 1]

        variance = statistics.variance([p1, p2])
        variances.append(variance)

    t = p_1_1 / (math.sqrt(1 / 5.0 * sum(variances)))
    if math.isnan(t):
        t = 0
    p_value = stats.t.sf(np.abs(t), 5) * 2.0

    return t, p_value
