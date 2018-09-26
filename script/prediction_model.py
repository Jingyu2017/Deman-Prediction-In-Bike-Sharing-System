#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 21:39:19 2018

@author: k1756990
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# =============================================================================
# multi-output Decision Tree Regressor (15min)
# =============================================================================

X = pd.read_csv('../output/finalInput.csv')
y = pd.read_csv('../output/finalInputLabel.csv')
y1 = y.iloc[:,:30] # 15 min
X_train, X_test, y_train, y_test = train_test_split(X, y1,
                                                    test_size=0.25,
                                                    random_state=2)

#The minimum number of samples required to split an internal node
gs = GridSearchCV(DecisionTreeRegressor(random_state=2),
                  param_grid={'min_samples_split': range(2, 503, 20)},
                  cv=5)
gs.fit(X_train, y_train)
results = gs.cv_results_
import matplotlib
plt.figure(figsize=(8, 5))
matplotlib.rcParams.update({'font.size': 18})
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.grid()
ax = plt.axes()
ax.set_xlim(0,501)
ax.set_ylim(0.63, 1)
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

for sample, style in (('train', '--'), ('test', '-')):
    sample_score_mean = results['mean_%s_score' % (sample)]
    sample_score_std = results['std_%s_score' % (sample)]
    ax.plot(X_axis, sample_score_mean, style, label="%s" % (sample))

    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    best_score = results['mean_test_score'][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.02))

plt.legend(loc="best")
plt.grid('off')
plt.show()

# =============================================================================
# ann (15min)
# =============================================================================
X = pd.read_csv('../output/finalInput.csv')
X = StandardScaler().fit_transform(X)
y = pd.read_csv('../output/finalInputLabel.csv')
y1 = y.iloc[:,:30] # 15 min
X_train, X_test, y_train, y_test = train_test_split(X, y1,
                                                    test_size=0.25,
                                                    random_state=2)
from sklearn.neural_network import MLPRegressor

gs = GridSearchCV(MLPRegressor(random_state=2),
                  param_grid={'hidden_layer_sizes': [(x,) for x in range(40, 303, 10)]},
                  cv=5)
gs.fit(X_train, y_train)
results = gs.cv_results_


import matplotlib
plt.figure(figsize=(8, 5))
plt.xlabel("hidden_layer_sizes")
plt.ylabel("Score")
plt.grid()
matplotlib.rcParams.update({'font.size': 18})
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
ax = plt.axes()
ax.set_xlim(40,303)
ax.set_ylim(0.7,0.9)
X_axis = range(40, 303, 10)

for sample, style in (('train', '--'), ('test', '-')):
    sample_score_mean = results['mean_%s_score' % (sample)]
    sample_score_std = results['std_%s_score' % (sample)]
    ax.plot(X_axis, sample_score_mean, style, label="%s" % (sample))

    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    best_score = results['mean_test_score'][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.02))

plt.legend(loc="best")
plt.grid('off')
plt.show()