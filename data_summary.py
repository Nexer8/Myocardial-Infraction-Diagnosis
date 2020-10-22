import os

import pandas as pd
import matplotlib.pyplot as plt

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
