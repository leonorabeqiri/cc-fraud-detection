#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


df = pd.read_csv("credit_card_dataset_with_smote.csv")


# In[36]:


df.shape


# In[37]:


from sklearn import preprocessing


# In[38]:


label_encoder = preprocessing.LabelEncoder()


# In[39]:


df['merchant']= label_encoder.fit_transform(df['merchant'])
df['merchant'].unique()


# In[40]:


df['category']= label_encoder.fit_transform(df['category'])
df['category'].unique()


# In[41]:


df['job']= label_encoder.fit_transform(df['job'])
df['job'].unique()


# In[42]:


df['gender']= label_encoder.fit_transform(df['gender'])
df['gender'].unique()


# In[43]:


df['city']= label_encoder.fit_transform(df['city'])
df['city'].unique()


# In[44]:


X = df.drop(columns=["is_fraud", "dob"], axis=1)
Y = df["is_fraud"] 


# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# In[46]:


classifier = GaussianNB()


# In[47]:


classifier.fit(X_train, Y_train)


# In[48]:


classifier.score(X_train, Y_train)


# In[49]:


Y_predicted=classifier.predict(X_test)


# In[50]:


cm=confusion_matrix(Y_test, Y_predicted)
cm


# In[51]:


report = classification_report(Y_test, Y_predicted, digits=3)
print(report)


# In[ ]:





# In[ ]:




