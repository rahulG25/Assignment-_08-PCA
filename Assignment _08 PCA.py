#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[2]:


wine= pd.read_csv("wine.csv")


# In[3]:


print(wine.describe())
wine.head()


# In[4]:


wine['Type'].value_counts()


# In[5]:


wine


# In[6]:


wine['Type'].value_counts()


# In[7]:


Wine= wine.iloc[:,1:]
Wine


# In[8]:


Wine.shape


# In[9]:


Wine.info()


# In[10]:


# Converting data to numpy array
wine_ary=Wine.values
wine_ary


# In[12]:


# Normalizing the  numerical data
wine_norm=scale(wine_ary)
wine_norm


# PCA Implementation

# In[13]:


# Applying PCA Fit Transform to dataset
pca = PCA()
pca_values = pca.fit_transform(wine_norm)
pca_values


# In[14]:


# PCA Components matrix or convariance Matrix
pca.components_


# In[15]:


# The amount of variance that each PCA has
var = pca.explained_variance_ratio_
var


# In[16]:


# Cummulative variance of each PCA
Var = np.cumsum(np.round(var,decimals= 4)*100)
Var


# In[17]:


plt.plot(Var,color="blue")


# In[18]:


# Final Dataframe
final_df=pd.concat([wine['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df


# In[19]:


# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)


# In[20]:


sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type')


# In[21]:


pca_values[: ,0:1]


# In[22]:


x= pca_values[:,0:1]
y= pca_values[:,1:2]
plt.scatter(x,y)


# # Checking with other Clustering Algorithms

# Hierarchical Clustering

# In[23]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[24]:


# As we already have normalized data, create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))


# In[25]:


# Create Clusters
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters


# In[26]:


y=pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[27]:


# Adding clusters to dataset
wine2=wine.copy()
wine2['clustersid']=hclusters.labels_
wine2


# # K-Means Clustering

# In[28]:


from sklearn.cluster import KMeans


# In[29]:


# As we already have normalized data
# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion WCSS 
# random state can be anything from 0 to 42, but the same number to be used everytime,so that the results don't change.


# In[39]:


# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)


# In[40]:


# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# # Build Cluster algorithm using

# K-3

# In[41]:


# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(wine_norm)
clusters3


# In[42]:


clusters3.labels_


# In[43]:


# Assign clusters to the data set
wine3=wine.copy()
wine3['clusters3id']=clusters3.labels_
wine3


# In[44]:


wine3['clusters3id'].value_counts()

