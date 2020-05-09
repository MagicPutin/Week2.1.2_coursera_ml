import sklearn
import numpy as np
import sklearn.neighbors
from matplotlib.pyplot import plot
from sklearn import datasets
import matplotlib.pyplot as plt # it's for fun. u can delete it, if you wouldn't like to see plot

# data
data = sklearn.datasets.load_boston()

# scaling dataset
data['target'] = sklearn.preprocessing.scale(data['target'])
# generator of folding
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# main task
max_p = 0
max_score = -1
scores = [] # also for fun, can be deleted
for k in np.linspace(start=1.0, stop=10.0, num=200):
    k_neighbors_regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance',
                                                                  p=k)
    score = sklearn.model_selection.cross_val_score(estimator=k_neighbors_regressor, X=data['data'], y=data['target'],
                                                    cv=kf, scoring='neg_mean_squared_error').mean()
    scores.append(score)
    if score > max_score:
        max_score = score
        max_p = k
    print('p: ' + str(k) + ' score: ' + str(score))
with open('answers/answer1.txt', 'w') as ans:
    ans.write(str(max_p))

# fun
plt.plot(np.linspace(start=1.0, stop=10.0, num=200), scores)
plt.show()
