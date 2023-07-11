# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:10:54 2022

@author: Deepak SK
"""
#logistic regression

import os 
import pandas as pd

os.chdir('C:/Users/Deepak SK/Downloads')
data=pd.read_csv("titanic.csv")
print(data.head())
#spliting data into 4 in python(x train,x test,y train,y test)
#in R we only split the data into 2(trinn set,test set)
print(data.dtypes)
print(data.columns)
#drop the columns which is has less corr 
corr=data.corr()
data1=data.drop(["PassengerId","Pclass","Name","SibSp","Parch","Ticket","Fare","Cabin"],axis=1)
print(data1.columns)
#to find the na's in the data
data1.isnull().sum()
data1.Age.median()
#fill the median value in place of na
data1["Age"].fillna(data1["Age"].median(),inplace=True)
#to get the frequency of the categorical data in Embarked column 
data1["Embarked"].value_counts()
#fill the S value in the place of missing palce
data1["Embarked"].fillna('S',inplace=True)
#
y=data1[["Survived"]]
x=data1.drop(["Survived"],axis=1)
print(y.head())
#to convert the categorical values into numarical values
x=pd.get_dummies(x)
print(x)
#splitin the data into train and test sets
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
#applying logistic regression
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(xtrain,ytrain)
pvalue=lm.predict(xtest)
print(pvalue)
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,pvalue)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,pvalue)
 