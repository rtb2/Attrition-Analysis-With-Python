# Importing Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir("A:\\Data Science\\Python Project Metro")

#importing dataset
dataset = pd.read_csv('HR_comma_sep.csv')

#Cleaning the data
#Checking for missing values
dataset.isnull().any()

#getting a quick overview
dataset.head()

dataset.columns

#Exploring the data
dataset.shape

dataset.dtypes

#attrition rate
attrition_rate = dataset.left.value_counts()/len(dataset)*100
attrition_rate

#dataset description
dataset.describe()

#Summary for left and retained
dataset.groupby("left").mean()

#Correlation between attributes
# As the correlation can only be acheived for numerical variables, lets change sales and salary to dummy variables
# create dummy variables for the categorical features
sales_dummies = pd.get_dummies(dataset.sales, prefix="sales_").astype("int")
salary_dummies = pd.get_dummies(dataset.salary, prefix="salary_").astype("int")

# stack the individual dummy sets together
dataset1 = pd.concat([sales_dummies, salary_dummies, dataset.drop(["sales", "salary"], axis=1)],axis=1)
correlation= dataset1.corr()
correlation
correlation.unstack().sort_values(ascending=False).drop_duplicates()

#Correlation Heatmap
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation, linewidths=2, mask=mask, cmap="Blues", vmax=.3, square=True)
plt.show()

#Salary Vs Attrition
sns.countplot(x="salary", hue='left', data=dataset).set_title('Employee Salary & Attrition Distribution')
plt.show()

#Department Vs attrition
plt.figure(figsize=(15,4),)
sns.countplot(x='sales', data=dataset)
plt.xticks(rotation=-45)
plt.show()

plt.figure(figsize=(15,4),)
sns.countplot(x="sales", hue='left', data=dataset).set_title('Employee Department & Attrition Distribution')
plt.xticks(rotation=45)
plt.show()

# attrition vs Project count 
fig = plt.figure(figsize=(15,4),)
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'number_project'] , color='g',shade=True,label='Stayed')
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'number_project'] , color='r',shade=True, label='Left')
plot.set(xlabel='Number Of Projects', ylabel='Frequency')
plt.title('Project Count Distribution - Left V.S. Stayed')
plt.show()

#Attrition Vs Evaluation
fig = plt.figure(figsize=(15,4),)
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'last_evaluation'] , color='g',shade=True,label='Stayed')
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'last_evaluation'] , color='r',shade=True, label='Left')
plot.set(xlabel='Number Of Projects', ylabel='Frequency')
plt.title('Employee Evaluation Distribution - Left V.S. Stayed')
plt.show()

#Attrition Vs Average Monthly hours
fig = plt.figure(figsize=(15,4),)
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'average_montly_hours'] , color='g',shade=True,label='Stayed')
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='Left')
plot.set(xlabel='Number Of Projects', ylabel='Frequency')
plt.title('Average Monthly Hours Distribution - Left V.S. Stayed')
plt.show()

#Attrition Vs Satisfaction
fig = plt.figure(figsize=(15,4),)
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 0),'satisfaction_level'] , color='g',shade=True,label='Stayed')
plot=sns.kdeplot(dataset.loc[(dataset['left'] == 1),'satisfaction_level'] , color='r',shade=True, label='Left')
plot.set(xlabel='Number Of Projects', ylabel='Frequency')
plt.title('Satisfaction Level Distribution - Left V.S. Stayed')
plt.show()

#Attrition Vs time spend at company
plt.figure(figsize=(15,4),)
sns.countplot(x="time_spend_company", hue='left', data=dataset).set_title('Years at Company & Attrition Distribution')
plt.show()

#Satisfaction Vs Evaluation
plt.figure(figsize=(8,5),)
plt.scatter(x=dataset['satisfaction_level'], y=dataset['last_evaluation'], c = dataset['left'], marker='.', cmap='Accent')
plt.xlabel("Satisfaction")
plt.ylabel("Last Evaluation")
plt.title ("Attrition Clusters")
plt.show()
    