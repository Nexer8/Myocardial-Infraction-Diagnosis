import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from data_parser import append_disease_column


def show_distribution(datasets, features_names):
    appended_list = append_disease_column(datasets, features_names)
    # described = df.describe().round(3)
    # print(described)

    # Check for NA
    # print(df.isnull().sum().sum())

    # Correlation matrix
    # correlation_matrix = df.corr().abs()

    # s = correlation_matrix.unstack()
    # so = s.sort_values(kind="quicksort")
    # print(so)

    # df.plot(kind='scatter', x='Age', y='Pain location')
    appended_list.groupby(['disease'])['disease'].agg(['count']).plot.pie(y='count', autopct='%1.1f%%')
    plt.show()


def show_best_score_confusion_matrix():
    best_score_confusion_matrix =[[28, 12, 9, 8, 14],
     [7, 7, 8, 7, 5],
     [7, 5, 62, 23, 2],
     [5, 5, 12, 110, 0],
     [20, 4, 3, 0, 88]]
    df_cm = pd.DataFrame(best_score_confusion_matrix, range(5), range(5))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, vmax=150)  # font size
    plt.show()

show_best_score_confusion_matrix()