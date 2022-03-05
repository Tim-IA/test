# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 00:47:53 2022

@author: Tim_secure
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

url2='https://raw.githubusercontent.com/Tim-IA/test/main/spam.csv'
df=pd.read_csv(url2, encoding='latin-1')
df=df.rename(columns={'v1':'class','v2': 'message'})
df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
df.columns = [ 'class', 'message']


# prepare the data for training by converting eht text data into vector form
cv=CountVectorizer()
data=cv.fit_transform(df['message'])
data=data.toarray()
data=pd.DataFrame(data)

#Splitting Training and Test Set
target=df['class']
x_train,x_test,y_train,y_test=train_test_split(data,target)
#Since we have a very small dataset, we will train our model with all availabe data.


rf=RandomForestClassifier()

#Fitting model with trainig data
rf.fit(x_train, y_train)

# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))
tester = st.radio(
    "Would you like to test our spam detection engine",
    ('Yes', 'No'))
if tester =='Yes':
    st.write("# Spam detection engine")
    message_text = st.text_input("Enter your message : ")        
else:
     st.write(" ")
     
model = pickle.load(open('model.pkl','rb'))
def classify_message(model, text):
  label = model.predict([text])[0]
  spam_prob = model.predict_proba([text])
  return {'type': label, 'spam probability': spam_prob[0][1]}

if message_text != '':

	result = classify_message(model, message_text)

	st.write(result)

	
	explain_pred = st.button('Explain Predictions')

	if explain_pred:
		with st.spinner('Generating explanations'):
			class_names = ['ham', 'spam']
			explainer = LimeTextExplainer(class_names=class_names)
			exp = explainer.explain_instance(message_text, 
				model.predict_proba, num_features=10)
			components.html(exp.as_html(), height=800)
