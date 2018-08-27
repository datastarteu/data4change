# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:07:35 2018

@author: pablo
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data for developing the model
train = pd.read_csv("./data/train.csv", index_col=0)

# Calculate correlations
train.corr()


label = 'Made Donation in March 2007'


# Create heatmap
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(train.corr())
plt.show()


sns.pairplot(train, hue=label)
plt.show()



cols = [c for c in train.columns if c != label]
# Univariate analysis
for col in cols:
    train.groupby(label)[col].hist()
    plt.title("Distribution by " + col)
    plt.legend(['No Donor','Donor'])
    plt.show()
    
   
train['Month Ratio'] = train['Months since Last Donation']/train['Months since First Donation']
sns.pairplot(train, hue=label)
plt.show()



train['Diff'] = np.log(1+train['Months since First Donation']-train['Months since Last Donation'])
train.groupby(label)['Diff'].hist()

