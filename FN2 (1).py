#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("fake_news.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data = data.drop(['id'], axis=1)


# In[7]:


data = data.fillna('')


# In[8]:


data['content'] = data['author']+' '+data['text']


# In[9]:


data = data.drop(['title','author','text'], axis=1)


# In[10]:


data.head()


# In[11]:


data['content'] = data['content'].apply(lambda x:"".join(x.lower() for x in x.split()))


# In[12]:


data['content'] = data['content'].str.replace("[^\w\s]",'')


# In[13]:


import nltk
nltk.download('stopwords')


# In[ ]:





# In[14]:


from nltk.corpus import stopwords
from textblob import Word
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x:"".join(Word(word).lemmatize() for word in x.split()))


# In[15]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content'] = data['content'].apply(lambda x:"".join(Word(word).lemmatize() for word in x.split()))
data['content'].head()


# In[16]:


x = data[['content']]
y = data['label']


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=45, stratify=y)


# In[20]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[21]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['content'])
xtrain_tfidf = tfidf_vect.transform(x_train['content'])
xtest_tfidf = tfidf_vect.transform(x_test['content'])


# In[27]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
pclf = PassiveAggressiveClassifier()
pclf.fit(xtrain_tfidf, y_train)
predictions = pclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[28]:


print(metrics.confusion_matrix(y_test,predictions))


# In[ ]:


from sklearn.neural_network import MLPClassifier
mlpclf = MLPClassifier(hidden_layer_sizes=(256,64,16),
                      activation = 'relu',
                      solver = 'adam')
mlpclf.fit(xtrain_tfidf, y_train)
predictions = mlpclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[ ]:


print(metrics.confusion_matrix(y_test,predictions))


# In[ ]:


import pickle
pickle.dump(mlpclf, open("fakenews1.pkl", "wb"))


# In[ ]:




