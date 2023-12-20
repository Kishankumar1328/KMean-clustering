#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[29]:


x,y=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)


# In[30]:


Kmean=KMeans(n_clusters=4)
Kmean.fit(x)


# In[31]:


Kmean.cluster_centers_


# In[32]:


plt.scatter(x[:,0],x[:,1],c=Kmean.labels_,cmap="viridis")
plt.scatter(Kmean.cluster_centers_[:,0],Kmean.cluster_centers_[:,1],marker="*",s=400,color="black")


# In[35]:


df=pd.read_csv("E:\annual-number-of-deaths-by-cause.csv")


# In[ ]:




