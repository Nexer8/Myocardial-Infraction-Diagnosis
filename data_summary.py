import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from data_parser import append_disease_column


def show_distribution(datasets, features_names):
    appended_list = append_disease_column(datasets, features_names)

    appended_list.groupby(['disease'])['disease'].agg(['count']).plot.pie(y='count', autopct='%1.1f%%')
    plt.show()


def show_best_score_confusion_matrix(best_score_confusion_matrix: list, diseases_names: list):
    df_cm = pd.DataFrame(best_score_confusion_matrix, diseases_names, diseases_names)

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, vmax=150, fmt='g')
    plt.show()
