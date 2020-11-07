import numpy as np
import pandas as pd
import statistics
import math
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from data_parser import load_features_names, load_all_files, load_data, load_diseases_names
from data_summary import show_distribution


def main():
    datasets = load_all_files()
    features_names = load_features_names()
    diseases_names = load_diseases_names()

    # data = load_data(datasets)
    # data.print_classes_strength()
    # data.print_combined_strength()

    # show_distribution(datasets, features_names)

    features = np.concatenate([*datasets])
    classes = np.concatenate(
        [np.full(dataset.shape[0], diseases_names[index]) for (index, dataset) in enumerate(datasets)])

    scores, p_values = chi2(features, classes)

    features_with_values = pd.concat(
        [pd.DataFrame(features_names, columns=['Features']), pd.DataFrame(scores, columns=['Scores']),
         pd.DataFrame(p_values, columns=['P_values'])], axis=1)

    # print(features_with_values.sort_values('Scores', ascending=False).round(3))
    ordered_features = features.copy()
    for idx, feature_idx in enumerate(features_with_values.sort_values('Scores', ascending=False).index):
        ordered_features[:, idx:idx + 1] = features[:, feature_idx: feature_idx + 1]

    # print('==============================================================')

    alpha = 0.05
    for index, row in features_with_values.iterrows():
        p_value = row['P_values']
        if p_value > alpha:
            features_with_values.drop(index, inplace=True)

    # print(features_with_values.sort_values('Scores', ascending=False).round(3))

    best_parameters = dict()
    best_score = 0
    rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

    n_neighbors_variants = [1, 5, 10]
    metric_variants = ['euclidean', 'minkowski']
    best_score_confusion_matrix = None
    best_score_classification_report = None

    df_columns = ['n_features', 'n_neighbors', 'metric', 'scores', 'mean_accuracy']
    results_df = pd.DataFrame(columns=df_columns)

    # number_of_features = ordered_features.shape[1] + 1
    number_of_features = 4

    print('Training models. Please wait...')
    for n_features in range(1, number_of_features):  # could be up to 7 features (range until 8)
        for n_neighbors in n_neighbors_variants:
            for metric in metric_variants:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                current_iteration_scores = []

                for train, test in rskf.split(ordered_features[:, 0:n_features], classes):
                    knn.fit(ordered_features[:, 0:n_features][train], classes[train])
                    current_best_score = knn.score(ordered_features[:, 0:n_features][test], classes[test])
                    current_iteration_scores.append(current_best_score)

                    if current_best_score > best_score:
                        best_score = current_best_score
                        best_parameters = {'n_neighbors': n_neighbors, 'metric': metric, 'n_features': n_features}

                        y_pred = knn.predict(ordered_features[:, 0:n_features][test])
                        best_score_confusion_matrix = confusion_matrix(classes[test], y_pred=y_pred)
                        best_score_classification_report = classification_report(classes[test], y_pred=y_pred)
                results_df.loc[len(results_df)] = [n_features, n_neighbors, metric, current_iteration_scores,
                                                 np.array(current_iteration_scores).mean().round(3)]
                #print(
                #    f'Mean score for n_neighbors={n_neighbors},metric={metric},n_features={n_features}: {np.array(current_iteration_scores).mean().round(3)}')

    results_df = results_df.sort_values('mean_accuracy')
    print('Best mean models scoreboard:')
    for i, row in results_df.iterrows():
        print(
            f'Mean score for n_neighbors={row["n_neighbors"]},metric={row["metric"]},n_features={row["n_features"]}: {row["mean_accuracy"]}')

    #print("T_TEST_DF length: " + str(len(results_df)))
    print(f'\nBest score: {best_score}')
    print(f'Best parameters: metric - {best_parameters["metric"]}, n_neighbors - {best_parameters["n_neighbors"]}, '
          f'number of features - {best_parameters["n_features"]}')

    # Compare every model by pair
    # compare_every_model_paired(t_test_df)

    # Compare two best models (indexed from 0 - best model)
    #compare_two_best_models(0, 5, t_test_df)

    # Find the first statistically significant different model pair in terms of mean accuracy (t-test)
    find_best_statistically_significant_model(results_df)

    print(best_score_confusion_matrix)
    print(best_score_classification_report)


def compare_two_best_models(model_idx1, model_idx2, df):
    alfa = 0.05
    df = df.sort_values('mean_accuracy', ascending=False)
    row1 = df.iloc[model_idx1]
    row2 = df.iloc[model_idx2]
    t, pvalue = paired_5x2_ttest(row1, row2)
    print(
        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]}, accuracy={row1["mean_accuracy"]}] '
        f'with [k={row2["n_neighbors"]}, m={row2["metric"]}, f={row2["n_features"]}, accuracy={row2["mean_accuracy"]}]')
    print(f'\tt-statistic: {t}, p-value: {pvalue}, alfa: {alfa}')

    if pvalue < alfa:
        print('\tIt is statistically significant! Good!')
    else:
        print('\tIt is NOT statistically significant! BAD!')


def find_best_statistically_significant_model(df):
    alfa = 0.05
    df = df.sort_values('mean_accuracy', ascending=False)
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and i < j:
                t, pvalue = paired_5x2_ttest(row1, row2)
                if pvalue < alfa:
                    print(
                        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]}, accuracy={row1["mean_accuracy"]}] '
                        f'with [k={row2["n_neighbors"]}, m={row2["metric"]}, f={row2["n_features"]}, accuracy={row2["mean_accuracy"]}]')
                    print(f'\tt-statistic: {t}, p-value: {pvalue}, alfa: {alfa}')
                    return


def compare_every_model_paired(df):
    alfa = 0.05
    statistical_significant_pairs = 0
    statistical_insignificant_pairs = 0
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j and i < j:
                t, pvalue = paired_5x2_ttest(row1, row2)
                if pvalue < alfa:
                    statistical_significant_pairs = statistical_significant_pairs + 1
                    print(
                        f'Comparing [k={row1["n_neighbors"]}, m={row1["metric"]}, f={row1["n_features"]}] with [k={row2["n_neighbors"]}, m={row2["metric"]}, f={row2["n_features"]}]')
                    print(f'\tt-statistic: {t}, p-value: {pvalue}, alfa: {alfa} -> {pvalue} (p) < {alfa} (a)')
                else:
                    statistical_insignificant_pairs = statistical_insignificant_pairs + 1

    print(
        f'Statistical significant pairs = {statistical_significant_pairs}, statistical insignificant pairs = {statistical_insignificant_pairs}')


def paired_5x2_ttest(m1, m2):
    p_1_1 = m1['scores'][0] - m2['scores'][0]
    variances = []
    for i in range(0, 5 * 2, 2):  # 5 repeats of 2-fold CV
        p1 = m1['scores'][i] - m2['scores'][i]
        p2 = m1['scores'][i + 1] - m2['scores'][i + 1]

        variance = statistics.variance([p1, p2])
        variances.append(variance)

    t = p_1_1 / (math.sqrt(1 / 5.0 * sum(variances)))
    pvalue = stats.t.sf(np.abs(t), 5) * 2.0
    return t, pvalue


if __name__ == '__main__':
    main()
