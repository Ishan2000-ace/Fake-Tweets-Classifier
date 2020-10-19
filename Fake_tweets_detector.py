# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:09:58 2020

@author: Ishan Nilotpal
"""


import pandas as pd
import numpy as np
import nltk
import re
import pickle

df = pd.read_csv('train.csv')

## Keywords  and location have nan values and there is no way to replace them and nothing to do with tweets so it is dropped

df = df.drop(['id','keyword','location'],axis=1)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus = []

for i in range(len(df['text'])):
        review = re.sub('[^a-zA-Z]',' ',df['text'][i])
        review = review.lower()
        review = review.split()
        review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review) 

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x=tf.fit_transform(corpus).toarray()
pickle.dump(tf,open('transform.pkl','wb'))


y = df['target']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

classifier.predict(x_test)

filename =open('nlp_model.pkl','wb')
pickle.dump(classifier,filename)



              
                    
        

