import numpy as np 
import pandas as pd
import os
import zipfile as zf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


# 1) Load bike sharing dataset 

files = zf.ZipFile("archive.zip", 'r')
files.extractall('bike-sharing-dataset')
files.close()

raw = pd.read_csv('bike-sharing-dataset/hour.csv')
print(raw.describe())
print(raw.hist(figsize=(12,10)))

def generate_dummies(df, dummy_column):
    dummies = pd.get_dummies(df[dummy_column], prefix=dummy_column)
    df = pd.concat([df, dummies], axis=1)
    return df

X = pd.DataFrame.copy(raw)
dummy_columns = ["season", "yr", "mnth", "hr", "weekday", "weathersit"]
for dummy_column in dummy_columns:
    X = generate_dummies(X, dummy_column)

for dummy_column in dummy_columns:
    del X[dummy_column]


first_5_weeks = 5*7*24 # 3 weeks (7 days), 24 hours each day
X[:first_5_weeks].plot(x='dteday', y='cnt', figsize=(18, 5))

y = X['cnt']
del X['cnt']
del X['casual']
del X['registered']

## drop also the variables 'instant' and 'dteday' since they are irrelevant

del X['instant']
del X['dteday']

# 2) Split the data set into training set and testing set, with 80% as training set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 22, test_size = 0.2)


# 3) Random Forest Regression model

# Grid Search
regressor = RandomForestRegressor()
parameters = [{'n_estimators' : [150,200,250,300], 
               'max_features' : [1.0 ,'sqrt','log2']}]

grid_search = GridSearchCV(estimator = regressor, param_grid = parameters)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Random Forest Regression model
# Use the best parameters found from above to build the model

regressor = RandomForestRegressor(n_estimators = best_parameters['n_estimators'], 
                                  max_features = best_parameters['max_features']) 
regressor.fit(X_train,y_train)

# Predicting the values 
y_pred = regressor.predict(X_test) 

# Comparing predicted values with true values in testing set
mean_absolute_error(y_test, y_pred)

# Using k-fold cross validation to evaluate the performance of the model
accuracy = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10)
accuracy.mean()

# Relative importance of features 

feature_importance = regressor.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12,10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# # Playing with the model
regressor = RandomForestRegressor(n_estimators=1500, max_samples=0.5, max_features=0.5, max_depth=30)


regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test) 

# Comparing predicted values with true values in testing set

mean_absolute_error(y_test, y_pred)

# Using k-fold cross validation to evaluate the performance of the model

accuracy = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10)
accuracy.mean()
