%%
import numpy as np    #Load the numpy library for fast array computations
import pandas as pd   #Load the pandas data-analysis library
import matplotlib.pyplot as plt   #Load the pyplot visualization library
import seaborn as sns

%%
from sklearn import datasets
dataset = datasets.load_iris()

%%
data=pd.DataFrame(dataset['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])

%%
data['Species']=dataset['target']
data['Species']=data['Species'].apply(lambda x: dataset['target_names'][x])

%%
data.head()

%%
data.shape

%%
data.describe()

%%
data.info()

%%
data.isnull().sum()

%%
sns.pairplot(data)

%%
cor = data.corr()
sns.heatmap(cor, annot =True)

%%
sns.scatterplot(x="Petal length", y="Petal Width",hue="Species", data=data)
