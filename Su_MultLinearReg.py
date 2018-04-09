#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:22:06 2018

@author: Suma
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap (might not be needed if the library already takes care of it)
X = X[:,1:]

#Splitting training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

#fitting the Multiple Linear regression model to the training set (X_train and y_train)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results (Y_pred - vector of predictions)
y_pred = regressor.predict(X_test)

'''Optimizing the model using backward elimination
#-----------------------------------------------'''
import statsmodels.formula.api as sm
#y=b0x0+b1x1+b2x2+..... where x0 = 1. so add one column with 1 to the beginning
#adding the intercept
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

'''Building Optimal matrix of features that contains only the 
independent variables that have high imapact on the 
dependent variable(profit) or arestatistically significant'''

X_opt = X[:,[0,1,2,3,4,5]] #Original matrix
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#removing the predictor variable with the highest P value (step by step)
X_opt = X[:,[0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Backward Elimination with p-values only:(another method - automatic)
'''
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''

