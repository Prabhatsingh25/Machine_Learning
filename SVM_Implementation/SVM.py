
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[4]:


iris = load_iris()
dir(iris)


# In[6]:


iris.feature_names


# In[7]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)


# In[9]:


df.head()


# In[13]:


df['target'] = iris.target
X = df.drop(['target'],axis='columns')
X
y=df.target
y


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[30]:


len(X_train)


# In[31]:


len(X_test)


# In[33]:


from sklearn.svm import SVC
svm_model = SVC()


# In[34]:


svm_model.fit(X_train,y_train)


# In[35]:


svm_model.score(X_test,y_test)

