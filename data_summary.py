import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import DataFrame

from data_parser import append_disease_column


def show_distribution(datasets, features_names):
    appended_list = append_disease_column(datasets, features_names)

    appended_list.groupby(['disease'])['disease'].agg(['count']).plot.pie(y='count', autopct='%1.1f%%')
    plt.show()


def show_best_score_confusion_matrix(best_score_confusion_matrix: list, diseases_names: list):
    df_cm = pd.DataFrame(best_score_confusion_matrix, sorted(diseases_names), sorted(diseases_names))

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, vmax=150, fmt='g')
    plt.show()


def show_summarising_plots(results_df: DataFrame, metric_variants: list, n_neighbors_variants: list, number_of_features: int):
    results_df = results_df.sort_values('n_features')

    for metric in metric_variants:
        results_per_metric = results_df.loc[results_df['metric'] == metric]
        for n_neighbors in n_neighbors_variants:
            results_per_neighbors_variant = results_per_metric.loc[results_per_metric['n_neighbors'] == n_neighbors]
            plt.plot(results_per_neighbors_variant['n_features'], results_per_neighbors_variant['mean_accuracy'],
                     label=f'k={n_neighbors}', linestyle='--', marker='o')

        plt.xlabel('Number of features')
        plt.xticks(range(1, number_of_features))
        plt.tick_params(axis='x', pad=8)
        plt.ylabel('Mean accuracy')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        plt.title(f'{metric.capitalize()} metric')
        plt.show()

    best_k = None
    for metric in metric_variants:
        results_df = results_df.sort_values('mean_accuracy', ascending=False)
        best_accuracy_row_for_metric = results_df.loc[results_df['metric'] == metric].iloc[0]

        results_df = results_df.sort_values('n_features')
        best_k = best_accuracy_row_for_metric['n_neighbors']
        best_graph_for_metric = results_df.loc[(results_df['n_neighbors'] == best_k) & (results_df['metric'] == metric)]

        plt.plot(best_graph_for_metric['n_features'], best_graph_for_metric['mean_accuracy'],
                 label=f'metric={metric}', linestyle='--', marker='o')

    plt.xlabel('Number of features')
    plt.xticks(range(1, number_of_features))
    plt.tick_params(axis='x', pad=8)
    plt.ylabel('Mean accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.title(f'Metrics comparison when k is the best (k={best_k})')
    plt.show()
