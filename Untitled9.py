#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# In[17]:


data, _ = make_blobs(n_samples=300, centers=3, random_state=42)


# In[18]:


plt.scatter(data[:, 0], data[:, 1], s=100, cmap='viridis')
plt.title("Generated Data with 3 Clusters")


# In[31]:


kmeans=KMeans(n_clusters=3)
kmeans.fit(data,_)


# In[32]:


centers = kmeans.cluster_centers_
labels = kmeans.labels_


# In[33]:


import matplotlib.pyplot as plt


# Scatter plot for data points with different colors based on cluster labels
plt.scatter(data[:, 0], data[:, 1],c=labels, s=50, cmap='viridis')

# Scatter plot for cluster centers as red X markers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

# Show the plot
plt.show()


# In[ ]:




