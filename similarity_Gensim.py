#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import gensim
import spacy
nlp = spacy.load('en_core_web_lg')


# In[ ]:


data = pd.read('test.csv')
data.head()
ref_sentence= data.loc[data['id']==15, 'text'].iloc[0]
ref_sent_vec= nlp(ref_sentence)
all_docs= [nlp(row) for row in data['text']]
sims=[]
doc_id=[]
for i in range(len(all_docs)):
    sim= all_docs[i].similarity(ref_sent_vec)
    sims.append(sim)
    doc_id.append[i]
sims_docs = pd.Dataframe(list(zip(doc_id,sims)), columns = ['doc_id', 'sims'])
sims_docs_sorted= sims_docs.sort_values(by='sims', ascending =False)
top5_sims_docs = data.iloc[sims_docs_sorted['doc_id'][1:6]]
print(data[data["id"]==15]['text'].values)
top_sim_scores= pd.concat([top5_sims_docs,sims_docs_sorted['sims'][1:6]], axis =1)
    for (text, sim) in zip(top_sim_scores['text'], top_sim_scores['sims']):
        print("top 5 similiar senteces are:\n with a similarity score of (: .2f)\n".format(text,sim))

