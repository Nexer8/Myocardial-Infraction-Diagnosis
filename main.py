import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors': [1, 5, 10], 'metric': ('euclidean', 'minkowski')}
classifier = KNeighborsClassifier()
