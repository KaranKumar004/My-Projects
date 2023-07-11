# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:30:05 2022

@author: Deepak SK
"""

##Load the data
import os 
import pandas as pd
os.chdir('C:/Users/Deepak SK/Downloads')
df=pd.read_csv("customer_reviews.csv")
df
##nltk-natural language tool kit
#pip install vader_lexicon
#nltk.download("vader_lexicon")
##call the function to analyse sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analysis=SentimentIntensityAnalyzer()
##check sentiment of the first feedback
senti_analysis.polarity_scores(df.iloc[9,1])
print(df.iloc[9,1])
##check the sentiment of text
df["score"]=df["text"].apply(lambda x:senti_analysis.polarity_scores(x))
##extracting compound score
df["compound_score"]=df["score"].apply(lambda x:x["compound"])
print(df)
import numpy as np
df["positive_negative"]=df["compound_score"].apply(lambda x:np.where(x>0,"Positive","Negative"))
##count of negative and positive feedback
df["positive_negative"].value_counts()
positive_data=df.query("positive_negative=='Positive'")
print(positive_data)
