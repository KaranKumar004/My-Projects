#Churn Analysis with XGBoost 
import os
os.getcwd()
os.chdir("C:/Users/Deepak SK/Downloads")
os.getcwd()
import pandas as pd
mydata = pd.read_csv("Telecom_Data.csv")
mydata
mydata.head()
mydata.dtypes
mydata.sample(frac=0.05)
y=mydata[["churn"]]
x=mydata.iloc[:,0:20]
x=pd.get_dummies(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
#!pip install xgboost 
from Xgboost import XGBClassifier
model=XGBClassifier()
model.fit(xtrain, ytrain)
predicted_value=model.predict(xtest)
from sklearn.metrics import confusion_matrix
confusion_matrix (ytest, predicted_value)
from sklearn.metrics import classification_report 
classification_report(ytest, predicted_value)