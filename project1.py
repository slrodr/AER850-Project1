# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:09:15 2024

@author: Santiagp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

"""
Read data and convert to dataframe
"""
df = pd.read_csv('Project_1_Data.csv')  
df = df.dropna()
df = df.reset_index(drop=True)

"""
Visualizig the Data
"""
df.hist()
#Creating a 3D plot to visualize data
fig = plt.figure()
ax = fig.add_subplot ( projection='3d')

# Extract X, Y, Z values and Step values
X = df['X']
Y = df['Y']
Z = df['Z']
step = df['Step']

colors = plt.cm.tab20(np.linspace(0, 1, 20))   # Use tab20 colourmap 

# Plot the scatter points with unique colors for each Step
for steps in range(1, 14): 
    mask = step == steps
    # Mapping each colour to a step
    ax.scatter(X[mask], Y[mask], Z[mask], color=colors[steps - 1], label=f'Step {steps}', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(title='Step', bbox_to_anchor=(1.05, 1))
plt.show()

"""
Correlation Analysis
"""

corr_matrix = df.corr(method='pearson')
plt.figure()
sns.heatmap(np.absolute(corr_matrix))
plt.show()

independent = df.drop('Step',axis=1)
corr_matrix2 = independent.corr(method='pearson')
plt.figure()
sns.heatmap(np.absolute(corr_matrix2))
plt.show()
"""
Classification Model Development
"""
#Splitting the data
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter1.split(df, df['Step']):
    df_train = df.loc[train_index].reset_index(drop=True)
    df_test = df.loc[test_index].reset_index(drop=True)

#Variable selection
X_train = df_train.drop("Step", axis=1)
X_test = df_test.drop("Step", axis=1)
y_train = df_train["Step"]
y_test = df_test["Step"]

#Scaling
scaler1 = StandardScaler()
X_train_scaled = pd.DataFrame(scaler1.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler1.transform(X_test), columns=X_test.columns, index=X_test.index)

'''Train and Test KNN'''
knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],           
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree','kd_tree','brute'],         
    'metric': ['euclidean', 'manhattan', 'minkowski', 'cityblock', 'cosine', 'haversine']  
    }

