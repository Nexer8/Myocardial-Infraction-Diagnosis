import os

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join('data', 'concatenated_source_data_heart_disease.csv'), sep=",")

described = df.describe().round(3)
print(described)

#df.plot(kind='scatter', x='Age', y='Pain location')
df.groupby(['disease'])['disease'].agg(['count']).plot.pie(y='count')
plt.show()