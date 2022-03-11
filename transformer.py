#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import pipeline
classifier = pipeline("zero-shot-classification", device=0)


# In[3]:


candidate_labels = ['positive','negative','neutral']

classifier("I'm handsome", candidate_labels)


# In[8]:


import streamlit as st
import numpy as np

st.title('Feedback')

text = 'Type!'
candidate_labels = ["positive", "negative", "neutral"]
user_input = st.text_input("Text", text)
doc = classifier(user_input, candidate_labels)

x = np.array(doc['scores'])
b = doc['labels']

m = np.char.find(b, 'positive')
m = np.where(m == 0 )
n = np.char.find(b, 'negative')
n = np.where(n == 0 )
k = np.char.find(b, 'neutral')
k = np.where(k == 0 )

st.write('Positive:', round(x[m][0], 2))
st.write('Negative:', round(x[n][0], 2)) 
st.write('Neutral:', round(x[k][0], 2)) 


# In[ ]:




