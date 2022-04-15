#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import time
import math


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# # Exploratory Data Analysis

# In[2]:


new_names = ['ID','1','2','3','4','5','6','7','8','9','10',
             '11','12','13','14','15','16','17','18','19','20','21','22','23','Label']

df_1 = pd.read_csv(
    'D:/Network Intrusion 5G/BotNeTIoT-L01_label_NoDuplicates.csv', 
    names=new_names,           # Rename columns
    header=0,                  # Drop the existing header row
    usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],       # Read the first 5 columns
)


# In[3]:


# number of normal = 0, attack=1

print("Unique = ", df_1['Label'].nunique())
print(df_1['Label'].value_counts())


# In[4]:


df_1.shape


# In[5]:


df_1.describe


# In[6]:


#Finding categorical features if any
num_cols = df_1._get_numeric_data().columns
cate_cols = list(set(df_1.columns)-set(num_cols))
cate_cols


# In[7]:


# Finding NaN, o values, if any
df_1.isnull().sum()


# In[8]:


# Check for constant values or abnormalities!

plt.figure(figsize = (16, 25))
for i in range(24):
    temp_data = df_1.iloc[:,i]
    plt.subplot(10,3,i+1)
    plt.boxplot(temp_data)
    plt.title("Senor: "+ str(i+1))
plt.show()


# In[19]:


#Visualizing the labels
def bar_graph(feature):
    df_1[feature].value_counts().plot(kind="bar")
    
bar_graph('Label')


# In[20]:


#plotting density distribution of the features.

plt.figure(figsize = (16, 25))
for i,j in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]):
    temp = df_1.iloc[:, j]
    plt.subplot(10,3, i+1)
    sns.kdeplot(temp, legend = False)
    plt.title("Column: "+ str(j))
plt.show()


# In[21]:


#heatmap for correlation coefficient
# calculate correlation
df_corr = df_1.drop(columns=["ID"]).corr()

# correlation matrix
sns.set(font_scale=0.8)
plt.figure(figsize=(24,16))
sns.heatmap(df_corr, annot=True, fmt=".4f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("coolwarm", 200))

plt.figtext(.45, 0.9,'correlation matrix of train_1', fontsize=16, ha='center')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




