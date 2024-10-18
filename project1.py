# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:09:15 2024

@author: Santiagp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
ax.set_zlabel('Z', )
ax.set_title('3D Scatter Plot of Coordinates with Steps')
ax.legend(title='Step', bbox_to_anchor=(1.05, 1))
plt.show()
# Note: This might qualify as data snooping bias as the data has not been
#split between train and test yet, however, the project steps ask for data
# visualization before data is prepared for the model and it is helpful
# to at least know what the data in question represents.

