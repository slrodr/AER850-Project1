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
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score
from sklearn.svm import SVC

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

'''KNN'''
#Creating model with best parameters
knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [5, 10, 15, 20, 25, 30],           
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree','kd_tree','brute'],         
    'metric': ['euclidean', 'manhattan', 'minkowski', 'cityblock']  
    }
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train_scaled, y_train)
best_model_knn = grid_search_knn.best_estimator_
print("Best KNN Model: ", best_model_knn)
print('Best parameters: ', grid_search_knn.best_params_)

#Train and test
y_train_pred_knn = best_model_knn.predict(X_train_scaled)
y_test_pred_knn = best_model_knn.predict(X_test_scaled)
mae_train_knn = mean_absolute_error(y_train, y_train_pred_knn)
mae_test_knn = mean_absolute_error(y_test, y_test_pred_knn)
print(f"KNN - MAE (Train): {mae_train_knn}, MAE (Test): {mae_test_knn}")
acc_train_knn = accuracy_score(y_train, y_train_pred_knn)
acc_test_knn = accuracy_score(y_test, y_test_pred_knn)
print(f"KNN - Acc (Train): {acc_train_knn}, Acc (Test): {acc_test_knn}")
f1_train_knn = f1_score(y_train, y_train_pred_knn, average='macro')
f1_test_knn = f1_score(y_test, y_test_pred_knn, average='macro')
print(f"KNN - F1 (Train): {f1_train_knn}, F1 (Test): {f1_test_knn}")
prec_train_knn = precision_score(y_train, y_train_pred_knn, average='macro')
prec_test_knn = precision_score(y_test, y_test_pred_knn, average='macro')
print(f"KNN - Precision (Train): {prec_train_knn}, Precision (Test): {prec_test_knn}")

'''SVM'''
#Create Model with best parameters
svc = SVC()
param_grid_svc = {
    'C': [0.001, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train_scaled, y_train)
best_model_svc= grid_search_svc.best_estimator_
print("Best SVC Model: ", best_model_svc)
print('Best parameters: ', grid_search_svc.best_params_)

#Train and Test
y_train_pred_svc = best_model_svc.predict(X_train_scaled)
y_test_pred_svc = best_model_svc.predict(X_test_scaled)
mae_train_svc = mean_absolute_error(y_train, y_train_pred_svc)
mae_test_svc = mean_absolute_error(y_test, y_test_pred_svc)
print(f"SVC - MAE (Train): {mae_train_svc}, MAE (Test): {mae_test_svc}")
acc_train_svc = accuracy_score(y_train, y_train_pred_svc)
acc_test_svc = accuracy_score(y_test, y_test_pred_svc)
print(f"SVC - Acc (Train): {acc_train_svc}, Acc (Test): {acc_test_svc}")
f1_train_svc = f1_score(y_train, y_train_pred_svc, average='macro')
f1_test_svc = f1_score(y_test, y_test_pred_svc, average='macro')
print(f"SVC - F1 (Train): {f1_train_svc}, F1 (Test): {f1_test_svc}")
prec_train_svc = precision_score(y_train, y_train_pred_svc, average='macro')
prec_test_svc = precision_score(y_test, y_test_pred_svc, average='macro')
print(f"SVC - Precision (Train): {prec_train_svc}, Precision (Test): {prec_test_svc}")