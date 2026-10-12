#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello")


# In[2]:


import sklearn
print(sklearn.__version__)


# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.sin(x)


# In[6]:


plt.plot(x,y)
plt.show()


# In[ ]:




