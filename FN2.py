#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


data = pd.read_csv("fake_news.csv")
data.head()


# In[31]:


data.shape


# In[32]:


data.info()


# In[33]:


data.isna().sum()


# In[34]:


data = data.drop(['id'], axis=1)


# In[35]:


data = data.fillna('')


# In[36]:


data['content'] = data['author']+' '+data['text']


# In[37]:


data = data.drop(['title','author','text'], axis=1)


# In[38]:


data.head()


# In[39]:


data['content'] = data['content'].apply(lambda x:"".join(x.lower() for x in x.split()))


# In[40]:


data['content'] = data['content'].str.replace("[^\w\s]",'')


# In[41]:


import nltk
nltk.download('stopwords')


# In[ ]:





# In[42]:


from nltk.corpus import stopwords
from textblob import Word
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x:"".join(Word(word).lemmatize() for word in x.split()))


# In[43]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content'] = data['content'].apply(lambda x:"".join(Word(word).lemmatize() for word in x.split()))
data['content'].head()


# In[44]:


x = data[['content']]
y = data['label']


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[49]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=45, stratify=y)


# In[52]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[53]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['content'])
xtrain_tfidf = tfidf_vect.transform(x_train['content'])
xtest_tfidf = tfidf_vect.transform(x_test['content'])


# In[ ]:




