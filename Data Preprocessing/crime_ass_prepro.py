import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/Shwet/Downloads/crime_data.csv")
df

df.shape

df.duplicated().sum()

#no duplicate values

df.dtypes

a=['Murder','Rape']
df[a]=df[a].astype(int)
df.dtypes

#--------checking outliers---------------#

sns.boxplot(df.select_dtypes(include='number'))

# feature Rape has a outliers

from feature_engine.outliers import Winsorizer

win=Winsorizer(capping_method='iqr',fold=1.5,tail='both',variables=['Rape'])
df1=win.fit_transform(df[['Rape']])
sns.boxplot(df1['Rape'])

#----------droping 0 varience features-------------#

varience=df.select_dtypes(include='number') 
varience   

var=varience.var() 
var   
    
zero_var=var[var<1].index.to_list()
zero_var

# there is no featues having zero varience











