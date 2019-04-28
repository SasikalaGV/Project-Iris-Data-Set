#!/usr/bin/env python
# coding: utf-8

# # Research
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.
# 
# It was actually called as Anderson's Iris data set because Edgar Anderson who collected the data to quantify the morphologic variation of Iris flowers of three related species. In relation to the accuracy of the data it said that two types of species were collected from the same land, picked on the same day and mearusred at the same time by the same person with the same apparatus.
# 
# This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.
# 
# The dataset contains a set of 150 records under 5 attributes -
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. Species: 
# -- Setosa 
# -- Versicolour 
# -- Virginica
# 

# In[1]:

from IPython.display import Image
Image(filename="attachments\Iris.png", width=600, height=600)


# Importing the libaries for this project: Pandas, Numpy.
# 
# Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.
# 
# NumPy is the fundamental package for scientific computing with Python
# 
# HoloViews is an open-source Python library designed to make data analysis and visualization seamless and simple.
# 
# Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
# 
# I also used the Jupyter Notebook for this project. 

# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns



# # Data
# Import the Irisdataset.csv using the panda library and examine first 10 rows of dataset

# In[3]:


Iris_data = pd.read_csv('attachments/Irisdataset.csv')

Iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

#specify the number of rows to be showen here
Iris_data.head(10)


# # Discovering the Shape of the table

# In[4]:


Iris_data.shape


# # Figure out unique type of Iris flower and the size

# In[5]:


Iris_data['species'].unique()


# In[6]:


Iris_data['species'].value_counts()


# # Analysing the data using Min, Max, Mean, Median and Standard Deviation
# 

# In this section, I have included few functions from Computations / Descriptive Stats of DataFrame

# Get the minimum value of all the column in the data set

# In[7]:


Iris_data.min()


# Get the maximum value of all the column in the data set

# In[8]:


Iris_data.max()


# 
# 
# Get the mean value of all the column in the data set

# In[9]:


Iris_data.mean()


# Get the median value of all the column in the data set

# In[10]:


Iris_data.median()


# Get the standard deviation value of all the column in data set

# In[11]:


Iris_data.std()


# # Summary statistics of the Iris_data Dataframe 
# DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 

# Describing all columns of a DataFrame regardless of data type

# In[12]:


Iris_data.describe(include='all')


# Excluding object columns from a Dataframe description

# In[13]:


Iris_data.describe(exclude=[np.object])


# # Plotting quantities from a CSV file
# Using pandas integrating plotting tool we can plot a few quantities, separately for each species Setosa, Veriscolor and Virginica.
# 
# 

# In[14]:


import pandas
import matplotlib.pyplot as plt  

species_data = Iris_data.groupby('species')
species_data.boxplot(column=['sepal_length', 'sepal_width' , 'petal_length', 'petal_width'])

from pandas.tools import plotting

plotting.scatter_matrix(Iris_data[['sepal_length', 'sepal_width' , 'petal_length', 'petal_width']])

plt.figure(figsize=(4,3))

plt.show() 


# # Seaborn Boxplot
# 
# By using different data visualization techniques through Seaborn we can compare the data for each species. Here I have taken Boxplot to show the differences between each Sepal and Petal.
# 
# The boxplot shows the distribution of quantitative data in a way that facilitates comparitions between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, expect for points that are determined to be "outliers" using a method that is a function of the inter-quartile range.
# 
# By using the boxplot I have included the comparisions for the following items from the Iris data set 
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width 
# 

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# set style, palette colour and figure size
sns.set(style="ticks", palette="Reds", rc={'figure.figsize':(10,10)})

title="Compare the Sepal Length for the Species"

# Load data to boxplot
sns.boxplot(x="species", y="sepal_length", data=Iris_data)

# Set font size
plt.title(title, fontsize=20)

# Show the plot
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# set style, palette colour and figure size
sns.set(style="ticks", palette="Reds", rc={'figure.figsize':(10,10)})

title="Compare the Sepal Width for the Species"

# Load data to boxplot
sns.boxplot(x="species", y="sepal_width", data=Iris_data)

# Set font size
plt.title(title, fontsize=20)

# Show the plot
plt.show()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# set style, palette colour and figure size
sns.set(style="ticks", palette="Reds", rc={'figure.figsize':(10,10)})

title="Compare the Petal Length for the Species"

# Load data to boxplot
sns.boxplot(x="species", y="petal_length", data=Iris_data)

# Set font size
plt.title(title, fontsize=20)

# Show the plot
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# set style, palette colour and figure size
sns.set(style="ticks", palette="Reds", rc={'figure.figsize':(10,10)})

title="Compare the Petal Width for the Species"

# Load data to boxplot
sns.boxplot(x="species", y="petal_width", data=Iris_data)

# Set font size
plt.title(title, fontsize=20)

# Show the plot
plt.show()


# # Pairplot
# Looking for relationships between variables across multiple dimensions

# In[19]:



import seaborn as sns
# Scatter plots for the features and histograms
# custom markers also applied
sns.pairplot(Iris_data, hue="species", palette="Reds", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[20]:


#setting the background color
import seaborn as sns
sns.set(style="whitegrid")

sns.pairplot(Iris_data, hue="species", palette="Reds", diag_kind="kde", markers=["o", "s", "D"])

#Remove the top and right spines from plot
sns.despine()

import matplotlib.pyplot as plt
plt.show()


# In[21]:


# plotting regression and confidence intervals
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(Iris_data, hue="species", kind='reg', palette="Reds")

#Remove the top and right spines from plot
sns.despine()

#show plot

plt.show()

#strong relationship between petal length and petal width and petal length and sepal length


# In[22]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="Reds", rc={'figure.figsize':(12,10)})

# "Melt" the dataset
iris2 = pd.melt(Iris_data, "species", var_name="measurement")

# Draw a categorical scatterplot
sns.swarmplot(x="measurement", y="value", hue="species",palette="Reds", data=iris2)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[23]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="Reds", rc={'figure.figsize':(12,10)})


sns.violinplot(x="species", y="petal_length", palette="Reds", data=Iris_data)

#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[24]:


import seaborn as sns
#setting the background color and size of graph
sns.set(style="whitegrid", palette="Reds", rc={'figure.figsize':(12,10)})

sns.violinplot(x="species", y="petal_width", palette="Reds", data=Iris_data)
#Remove the top and right spines from plot
sns.despine()

#show plot
import matplotlib.pyplot as plt
plt.show()


# In[25]:


import matplotlib.pyplot as plt
Iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# # Machine Learning using scikit-learn
# 
# Machine Learning in Python. This is a tutorial I found online. 
# 

# In[26]:



# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# # References
# 
# Background info
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# https://ifcs.boku.ac.at/repository/data/iris/index.html
# 
# Read and access data from the csv file using Panda
# https://pythonspot.com/pandas-read-csv/
# 
# Computations / Descriptive Stats of DataFrame
# https://pandas.pydata.org/pandas-docs/stable/reference/frame.html
# 
# Summary Statistics
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
# 
# Plotting quantities
# https://cogmaster-stats.github.io/python-cogstats/auto_examples/plot_pandas.html
# 
# Seaborn Boxplot
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
# 
# https://www.datacamp.com/community/tutorials/seaborn-python-tutorial#show
# 
# https://tacaswell.github.io/matplotlib/examples/statistics/boxplot_color_demo.html
# 
# matlablib
# https://matplotlib.org/gallery/index.html
# 
# Machine learning
# https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
# 
# 
# Statistics in Python
# http://www.scipy-lectures.org/packages/statistics/index.html#statistics
# 
# numpy tutorial
# https://www.dataquest.io/blog/numpy-tutorial-python/
# 
# Docs
# https://pandas.pydata.org/pandas-docs
# 
# 
# Machine Learning Tutorial
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
# 
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# 
# https://theseus.fi/bitstream/handle/10024/64785/yang_yu.pdf?sequence=1
# 
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# 
# http://seaborn.pydata.org/examples/scatterplot_categorical.html
# 
# 

# 
