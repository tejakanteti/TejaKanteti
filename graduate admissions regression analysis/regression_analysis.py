# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:41:26 2019

@author: tejak
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset = pd.read_csv('Admission_predict.csv')
dataset = dataset.iloc[:,1:9]
X = dataset.iloc[:,0:7]
y = dataset.iloc[:,7]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_1 = SVR(kernel = 'rbf')
regressor_1.fit(X_train, y_train)

# Predicting a new result using SVR 
y_pred_SVR = regressor_1.predict(X_test)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_2 = DecisionTreeRegressor(random_state = 0)
regressor_2.fit(X_train, y_train)

# Predicting a new result using Decision tree
y_pred_Decision_Tree = regressor_2.predict(X_test)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_3 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_3.fit(X_train, y_train)

# Predicting a new result using Random Forest
y_pred_Random_Forest = regressor_3.predict(X_test)



