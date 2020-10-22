import os

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join('data', 'concatenated_source_data_heart_disease.csv'), sep=",")

described = df.describe().round(3)
print(described)

# Check for NA
print(df.isnull().sum().sum())

# Correlation matrix
c = df.corr().abs()

s = c.unstack()
so = s.sort_values(kind="quicksort")
print(so)

#df.plot(kind='scatter', x='Age', y='Pain location')
df.groupby(['disease'])['disease'].agg(['count']).plot.pie(y='count')
plt.show()