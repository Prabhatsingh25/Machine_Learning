
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


from sklearn.datasets import load_digits
digit = load_digits()
dir(digit)


# In[12]:


df = pd.DataFrame(digit.data)
df['target'] = digit.target
df


# In[15]:


from sklearn.model_selection import train_test_split
X = df.drop('target',axis='columns')
y= df.target


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[17]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[18]:


model.score(X_test,y_test)

