# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
Created on Wed Jan 16 09:31:44 2019

@author: SURAJ BHADHORIYA
"""
#import libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#load dataset
df=pd.read_csv("amazon_dataset.csv")
#remove punctuation
df['review']=df['review'].str.replace('[^\w\s]','')
df['review'].head()
print(df['review'])

#fillna values
df=df.fillna({'review':''})
#eliminate rating =3
df=df[df['rating'] !=3]
print(df['rating'])
df['sentimate']=df['rating'].apply(lambda rating:+1 if rating>3 else -1)
print(df['sentimate'])

x=df['review']
y=df['sentimate']
#split dataset
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#form bag of words
vectorizer=CountVectorizer(token_pattern=r'\b\w+\b')
print(vectorizer)
train_matrix=vectorizer.fit_transform(X_train)
test_matrix=vectorizer.transform(X_test)
print(train_matrix)
print(test_matrix)


# apply model
classify=LogisticRegression()
model=classify.fit(train_matrix,y_train)
#accuracy
accuracy=classify.score(test_matrix,y_test)
print("accuracy =",accuracy)


#pre-processing for pridiction
sample_test_data=X_test[10:13]
print(sample_test_data)
sample_y=y_test[10:13]
print(sample_y)
sample_test_matrix=vectorizer.transform(sample_test_data)
score_test=classify.score(sample_test_matrix,sample_y)
print(score_test)
sample_test_data_predict=X_test[20:30]
print(sample_test_data_predict)
sample_y_predict=y_test[20:30]
print(sample_y_predict)
sample_test_matrix_predict=vectorizer.transform(sample_test_data_predict)
#pridiction
pridict=classify.predict(sample_test_matrix_predict)
print(pridict)
