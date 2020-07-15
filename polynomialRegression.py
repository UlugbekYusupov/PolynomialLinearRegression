#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:39:21 2020

@author: ulugbekyusupov
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#Training the Polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, y)

#Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression result(for higher resolution and smoother curve)
#X_grid = np.arange(min(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid),1))
#plt.scatter(X, y, color='red')
#plt.plot(X_grid,linear_regressor2.predict(pf.fit_transform(X_grid),color='blue'))
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.show()

#Predicting a new result with Linear Regression
print(linear_regressor.predict([[6.5]]))

#Predicting a new result with Polynomial Linear Regression
print(linear_regressor2.predict(pf.fit_transform([[6.5]],)))
