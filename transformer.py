#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import pipeline
classifier = pipeline("zero-shot-classification", device=0)


# In[8]:
import streamlit as st
import numpy as np

st.title('Sentiment app')

text = 'Type!'
candidate_labels = ["positive", "negative", "neutral"]
user_input = st.text_input("Text", text)
doc = classifier(user_input, candidate_labels)

x = np.array(doc['scores'])
b = doc['labels']

positive = np.char.find(b, 'positive')
positive = np.where(m == 0 )

negative = np.char.find(b, 'negative')
negative = np.where(n == 0 )

neutral = np.char.find(b, 'neutral')
neutral = np.where(k == 0 )

st.write('Positive:', round(x[positive][0], 2))
st.write('Negative:', round(x[negative][0], 2)) 
st.write('Neutral:', round(x[neutral][0], 2)) 







