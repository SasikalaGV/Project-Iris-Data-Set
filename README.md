# Project-Iris-Data-Set

Name : Sasikala Varatharajan Student Number : G00376470


This repository contains my solution to the Project - 2019 [Irish Data set] for the module 
Programming and Scripting at GMIT
[See here for the instructions](https://github.com/ianmcloughlin/project-pands/raw/master/project.pdf) {https://github.com/ianmcloughlin/project-pands/raw/master/project.pdf}


## How to download this repository
1. Go to GitHub
2. Click the 'Clone or Download' button
3. Click on Download as Zip

## Introduction

In this project I have taken Iris Flower Data set to do the basic research for Data Analysis. 

## Dataset
A dataset is an accumulation of data. Most usually a dataset compares to the contents of a single database table, or a single statistical data matrix, where each column of the table represents a specific variable, and each row relates to a given member of the data set in question. The dataset records esteems for every one of the variables, for example, height and weight of an object,for every individual from the data set. Each value is known as a datum. They can be analysed through queries based on different criteria, or dimensions.

## Fisher's Iris Data
# 
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems" as an example of linear discriminant analysis.

It was actually called as Anderson's Iris data set because Edgar Anderson who collected the data to quantify the morphologic variation of Iris flowers of three related species. In relation to the accuracy of the data it said that two types of species were collected from the same land, picked on the same day and mearusred at the same time by the same person with the same apparatus.

This famous iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

The dataset contains a set of 150 records under 5 attributes -

1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. Species: 
    - Setosa 
    - Versicolour 
    - Virginica


![iris](/attachments/Iris.png)

## Libraries Used
Importing the libaries for this project: Pandas, Numpy, Holoviews.

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools.

NumPy is the fundamental package for scientific computing with Python


import pandas as pd

import numpy as np

import seaborn as sns


# # Data
Import the Irisdataset.csv using the panda library and examine first 10 rows of dataset

Iris_data = pd.read_csv('attachments/Irisdataset.csv')

Iris_data.columns = ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width', 'species']

# specify the number of rows to be showen here

Iris_data.head(10)

Out:

![Iris_data_Head](/attachments/Iris_head.png)

# Discovering the Shape of the table

Iris_data.shape

Out:(150, 5)

# Figure out unique type of Iris flower and the size

Iris_data['species'].unique()

Out:

array(['setosa', 'versicolor', 'virginica'], dtype=object)

Iris_data['species'].value_counts()

Out:

virginica     50

setosa        50

versicolor    50

Name: species, dtype: int64

# Analysing the data using Min, Max, Mean, Median and Standard Deviation

In this section, I have included few functions from Computations / Descriptive Stats of DataFrame

Get the minimum value of all the column in the data set

Iris_data.min()

Out:

sepal_length       4.3

sepal_width          2

petal_length         1

petal_width        0.1

species         setosa

dtype: object


Iris_data.max()

Out:

sepal_length          7.9

sepal_width           4.4

petal_length          6.9

petal_width           2.5

species         virginica

dtype: object


Iris_data.mean()

Out:

sepal_length    5.843333

sepal_width     3.057333

petal_length    3.758000

petal_width     1.199333

dtype: float64


Iris_data.median()

Out:

sepal_length    5.80

sepal_width     3.00

petal_length    4.35

petal_width     1.30

dtype: float64


Iris_data.std()

Out:

sepal_length    0.828066

sepal_width     0.435866

petal_length    1.765298

petal_width     0.762238

dtype: float64

# Summary statistics of the Iris_data Dataframe 
DataFrame.describe: Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.

# Describing all columns of a DataFrame regardless of data type

Iris_data.describe(include='all')

Out:

![Iris_Describe](/attachments/Iris_desc.png)

# Excluding object columns from a Dataframe description

Iris_data.describe(exclude=[np.object])

Out:

![Iris_Describe](/attachments/Iris_npObject.png)



# Plotting quantities from a CSV file
Using pandas integrating plotting tool we can plot a few quantities, separately for each species Setosa, Veriscolor and Virginica.


import pandas
import matplotlib.pyplot as plt  

species_data = Iris_data.groupby('species')
species_data.boxplot(column=['sepal_length', 'sepal_width' , 'petal_length', 'petal_width'])

from pandas.tools import plotting

plotting.scatter_matrix(Iris_data[['sepal_length', 'sepal_width' , 'petal_length', 'petal_width']])

plt.figure(figsize=(4,3))

plt.show() 

Out:


![Iris_Describe](/attachments/PQ1.png)


![Iris_Describe](/attachments/PQ2.png)


![Iris_Describe](/attachments/PQ3.png)


![Iris_Describe](/attachments/PQ4.png)


# Seaborn Boxplot
By using different data visualization techniques through Seaborn we can compare the data for each species. Here I have taken Boxplot to show the differences between each Sepal and Petal.
 
The boxplot shows the distribution of quantitative data in a way that facilitates comparitions between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, expect for points that are determined to be "outliers" using a method that is a function of the inter-quartile range.

By using the boxplot I have included the comparisions for the following items from the Iris data set 
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width 

Out:
Compare the Sepal Length for the Species

![Iris_Describe](/attachments/compSPL.png)

Compare the Sepal Width for the Species

![Iris_Describe](/attachments/compSPW.png)

Compare the Petal Length for the Species

![Iris_Describe](/attachments/compPLL.png)

Compare the Petal Width for the Species

![Iris_Describe](/attachments/compPLW.png)



# Pairplot
Looking for relationships between variables across multiple dimensions


PairPlots

![Iris_Describe](/attachments/pairplot1.png)


![Iris_Describe](/attachments/pairplot2.png)


![Iris_Describe](/attachments/pairplot3.png)

Swarmplot

![Iris_Describe](/attachments/swarmplot1.png)


ViolinePlots

![Iris_Describe](/attachments/violineplot1.png)


![Iris_Describe](/attachments/violineplot2.png)

# Plots using subplots

![Iris_Describe](/attachments/subplot1.png)

# Machine Learning
Machine learning, as a powerful approach to achieve Artificial Intelligence, has been widely used in pattern recognition, Nowadays, with the development of technology, pattern recognition has become an essential and important technique in the field of Artificial Intelligence. The pattern recognition can identify letters, images, voice or other objects and also can identify status, extent or other abstractions.

![Iris_Describe](/attachments/ML1.png)


![Iris_Describe](/attachments/ML2.png)

