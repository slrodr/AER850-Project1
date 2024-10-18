# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:09:15 2024

@author: Santiagp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Project_1_Data.csv')  #Read data from file and convert to dataframe
df = df.dropna()
df = df.reset_index(drop=True)


df.hist()
