import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from data_parser import load_features_names, load_all_files, load_data, load_diseases_names
from data_summary import show_distribution, show_best_score_confusion_matrix
from model_analysis import find_best_statistically_significant_model, compare_every_model_paired, \
    compare_two_best_models


def main():
    datasets = load_all_files()
    features_names = load_features_names()
    diseases_names = load_diseases_names()

    data = load_data(datasets)
    data.print_classes_strength()
    data.print_combined_strength()

    show_distribution(datasets, features_names)

    features = np.concatenate([*datasets])
    classes = np.concatenate(
        [np.full(dataset.shape[0], diseases_names[index]) for (index, dataset) in enumerate(datasets)])

    scores, p_values = chi2(features, classes)

    features_with_values = pd.concat(
        [pd.DataFrame(features_names, columns=['Features']), pd.DataFrame(scores, columns=['Scores']),
         pd.DataFrame(p_values, columns=['P_values'])], axis=1)

    print(features_with_values.sort_values('Scores', ascending=False).round(3))
    ordered_features = features.copy()
    for idx, feature_idx in enumerate(features_with_values.sort_values('Scores', ascending=False).index):
        ordered_features[:, idx:idx + 1] = features[:, feature_idx: feature_idx + 1]

    print('==============================================================')

    alpha = 0.05
    for index, row in features_with_values.iterrows():
        p_value = row['P_values']
        if p_value > alpha:
            features_with_values.drop(index, inplace=True)

    print(features_with_values.sort_values('Scores', ascending=False).round(3))

    rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=1)

    n_neighbors_variants = [1, 5, 10]
    metric_variants = ['euclidean', 'minkowski']

    df_columns = ['n_features', 'n_neighbors', 'metric', 'scores', 'mean_accuracy', 'mean_confusion_matrix']
    results_df = pd.DataFrame(columns=df_columns)

    number_of_features = ordered_features.shape[1] + 1  # or set to 8

    print('Training models. Please wait...')
    for n_features in range(1, number_of_features):
        for n_neighbors in n_neighbors_variants:
            for metric in metric_variants:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
                current_iteration_scores = []
                current_iteration_confusion_matrices = np.zeros(shape=(5, 5))
                number_of_iterations = 0

                for train, test in rskf.split(ordered_features[:, 0:n_features], classes):
                    knn.fit(ordered_features[:, 0:n_features][train], classes[train])
                    current_score = knn.score(ordered_features[:, 0:n_features][test], classes[test])
                    current_iteration_scores.append(current_score)

                    y_pred = knn.predict(ordered_features[:, 0:n_features][test])
                    current_confusion_matrix = confusion_matrix(classes[test], y_pred=y_pred)
                    current_iteration_confusion_matrices += current_confusion_matrix

                    number_of_iterations += 1

                results_df.loc[len(results_df)] = [n_features, n_neighbors, metric, current_iteration_scores,
                                                   np.array(current_iteration_scores).mean().round(3),
                                                   (current_iteration_confusion_matrices / number_of_iterations)]

    results_df = results_df.sort_values('mean_accuracy')
    print('Best mean models scoreboard:')
    for i, row in results_df.iterrows():
        print(
            f'Mean score for n_neighbors={row["n_neighbors"]}, metric={row["metric"]}, '
            f'n_features={row["n_features"]}: {row["mean_accuracy"]}')

    compare_every_model_paired(results_df)

    compare_two_best_models(results_df)

    find_best_statistically_significant_model(results_df)

    best_model_params = results_df.sort_values('mean_accuracy', ascending=False).iloc[0]
    print(f'\nBest score: {best_model_params["mean_accuracy"]}')
    print(f'Best parameters: metric - {best_model_params["metric"]}, n_neighbors - {best_model_params["n_neighbors"]}, '
          f'number of features - {best_model_params["n_features"]}')

    show_best_score_confusion_matrix(best_model_params['mean_confusion_matrix'], diseases_names)


if __name__ == '__main__':
    main()
