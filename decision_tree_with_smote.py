#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("credit_card_dataset_with_smote.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


from sklearn import preprocessing


# In[6]:


label_encoder = preprocessing.LabelEncoder()


# In[7]:


df['merchant']= label_encoder.fit_transform(df['merchant'])
df['merchant'].unique()


# In[8]:


df['category']= label_encoder.fit_transform(df['category'])
df['category'].unique()


# In[9]:


df['job']= label_encoder.fit_transform(df['job'])
df['job'].unique()


# In[10]:


df['gender']= label_encoder.fit_transform(df['gender'])
df['gender'].unique()


# In[11]:


df['city']= label_encoder.fit_transform(df['city'])
df['city'].unique()


# In[12]:


X = df.drop(columns=["is_fraud", "dob"], axis=1)
Y = df["is_fraud"] 


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# In[14]:


clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100, max_depth=3, min_samples_leaf=5)


# In[15]:


clf_entropy.fit(X_train, Y_train)


# In[16]:


clf_entropy.score(X_test, Y_test)


# In[17]:


Y_predicted=clf_entropy.predict(X_test)


# In[18]:


cm=confusion_matrix(Y_test, Y_predicted)
cm


# In[19]:


report = classification_report(Y_test, Y_predicted, digits=3)
print(report)


# In[ ]:




