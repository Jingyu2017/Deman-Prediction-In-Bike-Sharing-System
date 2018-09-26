#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:03:25 2018

@author: k1756990
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


X = pd.read_csv('../output/finalInput.csv')
y = pd.read_csv('../output/finalInputLabel.csv')
 
y1 = y.iloc[:,:30] # 15 min
y2 = y.iloc[:,30:60] # 30 min
y3 = y.iloc[:,60:90] # 120 min


# =============================================================================
# PART I Prediction models
# 15  ##########  multi-target RF
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y1,
                                                    test_size=0.3,
                                                    random_state=2)

mrf15 = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
mrf15.fit(X_train, y_train)

mrf15_pred = mrf15.predict(X_test)
print '15,multi-target RF'
print mrf15.score(X_test, y_test)   #0.820724153017
# =============================================================================
# 30  ##########  multi-target RF
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y2,
                                                    test_size=0.3,
                                                    random_state=2)

mrf30 = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
mrf30.fit(X_train, y_train)

mrf30_pred = mrf30.predict(X_test)
print '30,multi-target RF'
print mrf30.score(X_test, y_test)   #0.899567085001
# =============================================================================
# 120   ##########  multi-target RF
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y3,
                                                    test_size=0.3,
                                                    random_state=2)

mrf120 = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
mrf120.fit(X_train, y_train)
mrf120_pred = mrf120.predict(X_test)
print '120,multi-target RF'
print mrf120.score(X_test, y_test)   # 0.959633602109



with open('../output/Prediction/mrf15.csv', 'w') as f:
    pd.DataFrame(mrf15_pred).to_csv(f)

with open('../output/Prediction/mrf30.csv', 'w') as f:
    pd.DataFrame(mrf30_pred).to_csv(f)

with open('../output/Prediction/mrf120.csv', 'w') as f:
    pd.DataFrame(mrf120_pred).to_csv(f)
# =============================================================================
#  15 ##########  Multi-target DT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y1,
                                                    test_size=0.3,
                                                    random_state=2)

mdt15 = DecisionTreeRegressor(random_state=2,min_samples_split=200)
mdt15.fit(X_train, y_train)
mdt15_pred = mdt15.predict(X_test)
print '15,multi-target DT'
print mdt15.score(X_test, y_test)   #0.806621051744


# =============================================================================
#  60 ##########  Multi-target DT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y2,
                                                    test_size=0.3,
                                                    random_state=2)

mdt30 = DecisionTreeRegressor(random_state=2,min_samples_split=200)
mdt30.fit(X_train, y_train)
mdt30_pred = mdt30.predict(X_test)
print '60,multi-target DT'
print mdt30.score(X_test, y_test)   #0.887569440504




# =============================================================================
#  120 ##########  Multi-target DT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y3,
                                                    test_size=0.3,
                                                    random_state=2)

mdt120 = DecisionTreeRegressor(random_state=2,min_samples_split=200)
mdt120.fit(X_train, y_train)
mdt120_pred = mdt120.predict(X_test)
print '120,multi-target DT'
print mdt120.score(X_test, y_test) #0.946515829637


with open('../output/Prediction/y_test.csv', 'w') as f:
    pd.DataFrame(y_test).to_csv(f)

with open('../output/Prediction/mdt15.csv', 'w') as f:
    pd.DataFrame(mdt15_pred).to_csv(f)

with open('../output/Prediction/mdt30.csv', 'w') as f:
    pd.DataFrame(mdt30_pred).to_csv(f)

with open('../output/Prediction/mdt120.csv', 'w') as f:
    pd.DataFrame(mdt120_pred).to_csv(f)

# =============================================================================
#  15 ##########  single RF
# =============================================================================


result = pd.DataFrame()
for i in range(30):
    yi = y.iloc[:,i]
    X_train, X_test, y_train, y_test = train_test_split(X, yi,
                                                        test_size=0.3,
                                                        random_state=2)
    rf = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
    rf.fit(X_train, y_train)
    result['%s'%i] = rf.predict(X_test)
    print i
      
   


# =============================================================================
#  30 ##########  single RF
# =============================================================================

result = pd.DataFrame()
for i in range(30,60):
    yi = y.iloc[:,i]
    X_train, X_test, y_train, y_test = train_test_split(X, yi,
                                                        test_size=0.3,
                                                        random_state=2)
    rf = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
    rf.fit(X_train, y_train)
    result['%s'%i] = rf.predict(X_test)
    print i

# =============================================================================
#  30 ##########  single RF
# =============================================================================
result = pd.DataFrame()
for i in range(60,):
    yi = y.iloc[:,i]
    X_train, X_test, y_train, y_test = train_test_split(X, yi,
                                                        test_size=0.3,
                                                        random_state=2)
    rf = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
    rf.fit(X_train, y_train)
    result['%s'%i] = rf.predict(X_test)
    print i
#mdt15.tree_.node_count  #2053
#mdt30.tree_.node_count  #1389
#mdt120.tree_.node_count #891

# =============================================================================
# ann
# =============================================================================
from sklearn.neural_network import MLPRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y1,
                                                    test_size=0.3,
                                                    random_state=2)
nn15 = MLPRegressor(random_state=2,hidden_layer_sizes=270)
nn15.fit(X_train, y_train)
nn15_pred = nn15.predict(X_test)
with open('../output/Prediction/nn15.csv', 'w') as f:
    pd.DataFrame(nn15_pred).to_csv(f)
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y2,
                                                    test_size=0.3,
                                                    random_state=2)
nn30 = MLPRegressor(random_state=2,hidden_layer_sizes=270)
nn30.fit(X_train, y_train)
nn30_pred = nn30.predict(X_test)
with open('../output/Prediction/nn30.csv', 'w') as f:
    pd.DataFrame(nn30_pred).to_csv(f)

X_train, X_test, y_train, y_test = train_test_split(X, y3,
                                                    test_size=0.3,
                                                    random_state=2)

nn120 = MLPRegressor(random_state=2,hidden_layer_sizes=270)
nn120.fit(X_train, y_train)
nn120_pred = nn120.predict(X_test)
with open('../output/Prediction/nn120.csv', 'w') as f:
    pd.DataFrame(nn120_pred).to_csv(f)



# =============================================================================
#  PART II   Feature importances with forests of trees
# (refenrence: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
# =============================================================================


X_train, X_test, y_train, y_test = train_test_split(X, y2,
                                                    test_size=0.3,
                                                    random_state=2)

mrf120 = RandomForestRegressor(n_estimators=200, random_state=2,n_jobs=-1)
mrf120.fit(X_train, y_train)

importances = mrf120.feature_importances_


indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
font = {'family' : 'normal',
        'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)

plt.bar(range(10), importances[indices][:10],
       )
plt.xticks(range(10), [
           'Check-in number at c15 during $\Delta t$',Feature importances with forests of trees (refenrence: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
           'Time',
           'Check-out number at c15 during $\Delta t$',
           'Check-out number at c3 during $\Delta t$',
           'Check-out number at c11 during $\Delta t$',           
           'Check-in number at c3 during $\Delta t$',
           'Check-in number at c3 during $\Delta t-1$',
           'isWeekend',
           r'Check-out number at c14 during $\Delta t$',
           r'Check-out number at c23 during $\Delta t$'],rotation=90)
plt.xlim([-1, 10])
plt.title('Prediction window: 30 minutes')
plt.show()




for i in indices[:10]:
    print X.columns[i]








