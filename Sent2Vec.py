#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip3 install sent2vec
from scipy import spatial
from sent2vec.vectorizer import Vectorizer

sentences = ["this defect is about account summary of policy center",
            "defect which is logged in policy center are related to admin",
            "cue application is throwing an error","est defect"]
vectorizer = Vectorizer()
vectorizer.bert(sentences)
vector_bert = vectorizer.vectors
vector_bert

dist = spatial.distance.cosine(vector_bert[1], vector_bert[0])

