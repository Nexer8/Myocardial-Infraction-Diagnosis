import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
from sklearn.model_selection import train_test_split, GridSearchCV
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

    for idx, feature_idx in enumerate(features_with_values.sort_values('Scores', ascending=False).index):
        features[:, [feature_idx, idx]] = features[:, [idx, feature_idx]]

    # print('==============================================================')

    alpha = 0.05
    for index, row in features_with_values.iterrows():
        p_value = row['P_values']
        if p_value > alpha:
            features_with_values.drop(index, inplace=True)

    # print(features_with_values.sort_values('Scores', ascending=False).round(3))

    # X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.25, random_state=0) #
    # TODO: is it necessary to split for training and testing sets?

    param_grid = {'n_neighbors': [1, 5, 10], 'metric': ('euclidean', 'minkowski')}

    best_parameters = dict()
    best_score = 0
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=2, n_jobs=-1)

    for feature in range(1, features.shape[1] + 1):  # could be up to 7 features (range until 8)
        current_iteration_scores = []
        for idx in range(5):
            clf.fit(features[:, 0:feature], classes)
            current_best_score = clf.best_score_
            current_iteration_scores.append(current_best_score)

            if current_best_score > best_score:
                best_score = current_best_score
                best_parameters = clf.best_params_
                best_parameters['features'] = feature

        print(f'Mean score for {feature} features: {np.array(current_iteration_scores).mean().round(3)}')

        # if feature > 1:
        #     stats.ttest_rel(current_iteration_scores[feature - 1], current_iteration_scores[feature])
        #     TODO: add the t-student test

    print(f'\nBest score: {best_score}')
    print(f'Best parameters: metric - {best_parameters["metric"]}, n_neighbors - {best_parameters["n_neighbors"]}, '
          f'number of features - {best_parameters["features"]}')

    clf.fit(features[:, 0:best_parameters['features']], classes)
    y_pred = clf.best_estimator_.predict(features[:, 0:best_parameters['features']])
    print(confusion_matrix(classes, y_pred=y_pred))


if __name__ == '__main__':
    main()
