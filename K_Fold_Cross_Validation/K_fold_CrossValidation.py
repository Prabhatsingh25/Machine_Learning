
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
digit = load_digits()


# In[11]:


svm_model = SVC()


# In[12]:


lr_model = linear_model.LogisticRegression()


# In[13]:


rf_model = RandomForestClassifier()


# In[14]:


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits = 3)


# In[15]:


from sklearn.model_selection import cross_val_score
cross_val_score(svm_model,digit.data,digit.target)


# In[16]:


from sklearn.model_selection import cross_val_score
cross_val_score(lr_model,digit.data,digit.target)


# In[17]:


from sklearn.model_selection import cross_val_score
cross_val_score(rf_model,digit.data,digit.target)

