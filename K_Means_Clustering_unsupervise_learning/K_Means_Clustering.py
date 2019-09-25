
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("J:\Machine_Learning\K_Means_Clustering_unsupervise_learning\income.csv")
df.head()


# In[6]:


df = df.drop('Name',axis='columns')
df.head()


# In[8]:


plt.scatter(df.Age,df['Income($)'],color='blue')


# In[9]:


km = KMeans(n_clusters=3)                  # we are ourself defining that the cluster is 3 there


# In[10]:


y_predict = km.fit_predict(df)             # it will makr each point in the three cluster


# In[11]:


y_predict


# In[22]:


df['cluster'] = y_predict
df.head()


# In[23]:


df1 = df[df.cluster==0]                             # we create 3 dataframe for 0th cluster, 1st cluster and 2nd cluster
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='blue')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='green')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()                                        # responisble for upper left corner box defining colour or marker




                         # But when we draw the graph we find the cluster is not perfect as Green dot's are very much appart
                         # this is because of the scale in x and y axis, To solve it we must bring them in same scale for that
                         # we use "MinMaxScaller"


# In[24]:


scaler = MinMaxScaler()                               # To bring the scale of x and y axis at same range 

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
df.head()

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df.head()


# In[28]:


df= df.drop('cluster',axis='columns')   # After changing the scale we againg run kMean clustering
km = KMeans(n_clusters=3)
cluster = km.fit_predict(df)
cluster


# In[29]:


df['cluster'] = cluster
df.head()


# In[32]:


km.cluster_centers_                          # This will give the x and y axis of all the centroid


# In[37]:


df1 = df[cluster==0]
df2 = df[cluster==1]
df3 = df[cluster==2]

plt.scatter(df1.Age,df1['Income($)'], color='blue')
plt.scatter(df2.Age,df2['Income($)'], color='red')
plt.scatter(df3.Age,df3['Income($)'], color='green')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*')

plt.xlabel = 'Age'
plt.ylabel = 'Income($)'

plt.legend()

                # Again draw the graph and this time every thing is fine


# In[40]:


# But in above we define cluster our own without thinking that the algo might work better with some more
# or some less cluster.
              
# For that we have a Algo called 'Elbo' which give us a better choice of cluster By finding SSE 
# (Sum of Square Error)




sse =[]                            # this is a way of finding Sum of Square Error
k_range = range(1,11)
for i in k_range:
    km = KMeans(n_clusters=i)
    km.fit(df)
    sse.append(km.inertia_)        # 'inertia' give us the SSE for each 'K'
    


# In[41]:


sse                               # these are the different SSE


# In[45]:


# Now we plot the 'SSE' with respet to 'K' to find the 'Elbo'.


plt.xlabel = 'K'
plt.ylabel = 'sse'
plt.plot(k_range,sse)


                            # And we find at 3 we feel like Elbo.

