# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:13:38 2022

@author: Deepak SK
"""

#computer price using ramdom_forest
import os
os.getcwd()
os.chdir("C:/Users/Deepak SK/Downloads")
os.getcwd()
import pandas as pd
mydata = pd.read_csv("Computer_Data (1).csv")
mydata.head()
#ther is no missing values
mydata.isnull().sum()
mydata.info()
A=mydata.corr()
print(mydata.columns)
#removing the columns ads,unnamed and trend has their correlation is low
newdata=mydata.drop(["ads","Unnamed: 0","trend"],axis=1)
newdata
n=pd.get_dummies(newdata.cd)
x=newdata.drop(['price'],axis=1)
y=newdata.price
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)
xtrain.shape
xtest.shape
xtrain=pd.get_dummies(xtrain)
xtest=pd.get_dummies(xtest)

from sklearn.ensemble import RandomForestRegressor
lm=RandomForestRegressor(random_state=0)
lm.fit(xtrain,ytrain)
pvalue=lm.predict(xtest)
print(pvalue)
accuracy = lm.score(xtest,ytest)
accuracy
