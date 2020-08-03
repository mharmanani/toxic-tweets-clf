#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Natural Language Processing
import nltk, re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[6]:


STOPWORDS = ['the', 'i', 'a'] # define different stop words


# In[7]:


df = pd.read_csv('../data/train.csv')


# In[8]:


df.loc[df['label'] == 1].head()


# In[9]:


def is_hashtag(word : str) -> bool:
    return word[0] == '#'
def is_mention(word: str) -> bool:
    return word[0] == '@'


# In[10]:


def clean_tweet(tweet : str) -> str:
    """ 
    Remove/replace mentions/hashtags with words.
    """
    tokens = tweet.split()
    for token in tokens:
        token = token.strip()
        if is_hashtag(token):
            tweet=tweet.replace(token, token[1:])
            #tweet=tweet.replace(token, '')
        if is_mention(token):
            tweet=tweet.replace(token, '')
    return tweet


# In[11]:


df['tweet']=df['tweet'].apply(lambda text : ''.join([char for char in text if char not in string.punctuation]))
df['tweet']=df['tweet'].apply(clean_tweet)


# In[12]:


df.head()


# In[13]:


df['label'].hist()


# In[14]:


df['tweet']=df['tweet'].apply(nltk.word_tokenize)
df['tweet']=df['tweet'].apply(lambda x: list(filter(lambda y: not y in STOPWORDS, x)))
df['tweet']=df['tweet'].apply(lambda x: list(filter(lambda y: y.isalpha(), x)))


# In[15]:


df.loc[df['label'] == 1].tail()


# In[16]:


# Predicting Parts of Speech for Each Token
def tag_parts_of_speech(token):
    return nltk.pos_tag(nltk.word_tokenize(token))

def pos_tagger(tokens):
    ts = []
    for word in tokens:
        ts += tag_parts_of_speech(word)
    return ts


# In[17]:


lemmatizer = WordNetLemmatizer()


# In[18]:


df['tweet']=df['tweet'].apply(lambda xs: [lemmatizer.lemmatize(x) for x in xs])


# In[19]:


df.head()


# In[20]:


df.tweet=df.tweet.apply(lambda x: ' '.join(x))


# In[21]:


# Model Building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# Testing
from sklearn.model_selection import train_test_split


# In[22]:


ngram_cv = CountVectorizer(binary=True, ngram_range=(3,5))


# In[23]:


X = df.tweet
y = np.array(df.label, dtype="float64")


# In[24]:


X_train, X_val, y_train, y_val = train_test_split(X, y)


# In[25]:


ngram_cv.fit(X_train, y_train)


# In[26]:


X_train = ngram_cv.transform(X_train)
X_val =  ngram_cv.transform(X_val)


# In[27]:


c = 0.075
lr = LogisticRegression(C=c, solver='lbfgs')
svm = LinearSVC(C=c)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                   hidden_layer_sizes=(5, 2), random_state=1)


# In[28]:


lr.fit( X_train, y_train)
svm.fit(X_train, y_train)
nn.fit(X_train, y_train)
pass


# In[31]:


def model_accuracy(model):
    count, total = 0, 0
    for x in range(X_val.shape[0]):
        pair = X_val[x], y_val[x]
        if model.predict(pair[0]) == pair[1]:
            count += 1
        total += 1
    print(round(100*count/total, 2), end='%')


# In[32]:


model_accuracy(lr)


# In[33]:


model_accuracy(svm)


# In[34]:


model_accuracy(nn)


# In[55]:


df_test = pd.read_csv("../data/test.csv")


# In[56]:


df_test['orig_tweet']=df_test['tweet'].apply(lambda x: x)
df_test['tweet']=df_test['tweet'].apply(clean_tweet)
df_test['tweet']=df_test['tweet'].apply(nltk.word_tokenize)
df_test['tweet']=df_test['tweet'].apply(lambda x: list(filter(lambda y: not y in STOPWORDS, x)))
df_test['tweet']=df_test['tweet'].apply(lambda x: list(filter(lambda y: y.isalpha(), x)))
df_test['tweet']=df_test['tweet'].apply(lambda xs: [lemmatizer.lemmatize(x) for x in xs])
df_test.tweet=df_test.tweet.apply(lambda x: ' '.join(x))


# In[57]:


X_example = ngram_cv.transform(df_test.tweet)


# In[58]:


def predict(model):
    for x in range(X_example.shape[0]):
        i, p = df_test.tweet[x], model.predict(X_example[x])
        if p:
            print(i ,p)


# In[59]:


def predict_message(model, message):
    return model.predict(ngram_cv.transform(pd.DataFrame({'tweet': [message]}).tweet))


# In[61]:


df_test['label'] = df_test['tweet'].map(lambda tweet: 0)
df_test['label'] = df_test['tweet'].map(lambda tweet: int(predict_message(svm, tweet)))


# In[63]:


ones = df_test[df_test.label == 1] # shows examples of tweets classified as toxic


# In[65]:


ones[['orig_tweet', 'label']]

