#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:23:33 2017

@author: cyanide
"""
"""
Problem Statement
To check if the employee at level 6
earning the salary mentioned in the data who claims to be
halway through to becoming a level 7 employee is telling the truth or bluffing.
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# We need X to be a matrix even when it has a single column of data in it
# As an alternative, we can use X.reshape(len(X), 1) to convert vector X to a matrix
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Since the dataset given is too small
# and we need as accurate a predicion as possible,
# so we let go spltting just for this case.

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Starting from degree = 2, we increment the degree gradually until we get the best fit curve
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression Results
plt.scatter(X, y , color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results
plt.scatter(X, y , color = 'red')
# Remember to fit the features X to a PolynomialFeatures object before prediction
# We could use X_poly here but to generalize it
# for all future test sets we use the fit_transform() function
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# To get a smoother curve, we need to plot
# a number of intermediate points at equal steps
# Here, the distance covered in each step = 0.01
X_grid = np.arange(min(X), max(X), 0.1)
# X_grid is a vector and needs to be converted to a matrix
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y , color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Poynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

"""Since the prediction is accurate, thus the employee is telling the truth"""
