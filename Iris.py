#!/usr/bin/env python
# coding: utf-8

# In[271]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[272]:


iris =pd.read_csv(r'C:\\Users\\Prabhat Singh\Desktop\dataset\Iris.csv') 


# In[273]:


iris.head()


# In[274]:


iris.drop(["Id"],axis=1,inplace=True)


# In[275]:


iris.head()


# In[276]:


sns.heatmap(data=iris.corr(),annot=True)
plt.show()


# In[277]:


iris.hist(figsize=(35,20),edgecolor="black")


# In[278]:


# Importing all the Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[279]:


train, test = train_test_split(iris,test_size=0.3)
print("The shape of test: {}". format(test.shape))
print("The shape of train: {}". format(train.shape))


# In[280]:


test_X = test[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
train_X = train[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
test_y = test.Species
train_y = train.Species


# In[281]:


le = LabelEncoder()
train_y = le.fit_transform(train_y)
test_y = le.fit_transform(test_y)


# In[282]:


#Logictic Regression
lr = LogisticRegression()
lr.fit(train_X,train_y)
pred_lr = lr.predict(test_X)
print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_lr,test_y)*100))


# In[283]:


# Random Forest Regression
rfr = RandomForestRegressor()
rfr = rfr.fit(train_X,train_y)
pred_rfr = rfr.predict(test_X)
print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_rfr.round(),test_y)*100))


# In[284]:


# support vector machine
sv_m = SVR()
sv_m =sv_m.fit(train_X,train_y)
pred_svm = sv_m.predict(test_X)
print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_svm.round(),test_y)*100))



# In[286]:


dtc = DecisionTreeClassifier()
dtc.fit(train_X,train_y)
pred_dtc = dtc.predict(test_X)
print("The accuracy of the model:{}".format(metrics.accuracy_score(pred_dtc,test_y)*100))


# In[ ]:




