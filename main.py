import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier

from data_parser import load_features_names, load_all_files, load_data, to_dataframe, save_dataframe_to_file

parameters = {'n_neighbors': [1, 5, 10], 'metric': ('euclidean', 'minkowski')}
classifier = KNeighborsClassifier()




def main():
    datasets = load_all_files()
    features_names = load_features_names()

    data = load_data(datasets)
    data.print_classes_strength()

    df = to_dataframe(datasets, features_names)
    save_dataframe_to_file(df)

    features = np.concatenate([*datasets])
    classes = np.concatenate(
        [np.full(dataset.shape[0], index) for (index, dataset) in enumerate(datasets)])  # nazwa pliku zamiast index

    scores, p_values = chi2(features, classes)

    features_with_score_values = pd.concat(
        [pd.DataFrame(features_names, columns=['Features']), pd.DataFrame(scores, columns=['Scores']),
         pd.DataFrame(p_values, columns=['P_values'])], axis=1)
    features_with_score_values.index += 1  # To match with feature's number

    print(features_with_score_values.sort_values('Scores', ascending=False).round(3))

    print('==============================================================')

    alpha = 0.05
    for index, row in features_with_score_values.iterrows():
        p_value = row['P_values']
        if p_value > alpha:
            features_with_score_values.drop(index, inplace=True)

    print(features_with_score_values.sort_values('Scores', ascending=False).round(3))


if __name__ == '__main__':
    main()
