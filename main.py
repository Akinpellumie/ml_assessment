# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score, RepeatedStratifiedKFold,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

import plotly 
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

#import missingno as msno

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("fetal_health.csv")
df
df.info
df.describe().T

#display by default first 5 rows of dataset
df.head()
print("missing values by column")
print("-"*30)
print(df.isna().sum())
print("-"*30)
print("TOTAL MISSINNG VALUES:",df.isna().sum().sum())
print (f' We have {df.shape[0]} instances with the {df.shape[1]-1} features and 1 output variable')

print(df.describe())
plt.matshow(df.corr())
plt.show()
df.duplicated().sum()
df.info()

y = df['fetal_health']
print(f'Percentage of normal fetal health: % {round(y.value_counts(normalize=True)[1]*100,2)} --> ({y.value_counts()[1]} patient)\nPercentage of suspected fetal health issue: % {round(y.value_counts(normalize=True)[2]*100,2)} --> ({y.value_counts()[2]} patient)\nPercentage of pathological fetal health issue: % {round(y.value_counts(normalize=True)[3]*100,2)} --> ({y.value_counts()[3]} patient)')

fig = px.histogram(df, x="fetal_health", title='Fetal Health', width=400, height=400)
fig.show()
df_target=df['fetal_health']
df_feature=df.drop('fetal_health',axis=1)


