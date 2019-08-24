#!/usr/bin/env python
# coding: utf-8

# <h1><center>WORD EMBEDDINGS FOR TEXTUAL SIMILARITY</center></h1>
# <h2><center>BIKASH GYAWALI</center></h2>
# <h1><center>bikashg@live.com</center></h1>

#   
NLP deals with text processing at various levels of representation. 


To understand the meaning of a text (sentence) is to understand the meaning of its words -- useful for many NLP applications like IR, QA, SA etc.



"You shall know a word by the company it keeps" (John Firth)



I play __  : play/read/eat, NP (football/guitar/books but not France)



Word embeddings are techniques/means to encode words via numbers (real valued vectors. Eg: play = [0.07,-0.5,0.98]) such that similar words get similar representation. 

This presentation is about using word embeddings for specific NLP task (determining textual similarity). We will not tak about how to build word embedding models.
# <img src="wordEmbedding_IO.jpg">

#   

# <img src="word-embeddings.png">

# In[1]:


# Image copied from https://www.adityathakker.com/introduction-to-word2vec-how-it-works/


# In[ ]:





# # 1. Word2Vec

# In[2]:


import gensim

import numpy as np

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)


# In[3]:


vec_computer = word2vecModel.word_vec("computer")


# In[4]:


print(vec_computer)


# In[5]:


print(len(vec_computer))


# In[4]:


vec_website = word2vecModel.word_vec("website")


# In[5]:


print(len(vec_website))


# In[6]:


vec_orange = word2vecModel.word_vec("orange")


# In[7]:


print(len(vec_orange))


# In[22]:


vec_oov = word2vecModel.word_vec("embeddings")


# ## Compute similarity of words
# 
# The cosine similarity value can range from -1 (exactly opposite) to 1 (exactly the same). See https://en.wikipedia.org/wiki/Cosine_similarity

# In[8]:


similarity_computer_website = word2vecModel.similarity('computer', 'website')
print(similarity_computer_website)


# In[9]:


similarity_orange_website = word2vecModel.similarity('orange', 'website')
print(similarity_orange_website)


# In[10]:


def compute_cosine_similarity(a,b):
#     Copied from https://skipperkongen.dk/2018/09/19/cosine-similarity-in-python/
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos



print(compute_cosine_similarity(vec_computer,vec_website))
print(compute_cosine_similarity(vec_orange,vec_website))


# ## Demo
# http://bionlp-www.utu.fi/wv_demo/

# In[ ]:





# In[ ]:





# In[ ]:





# # 2. BERT Embeddings
# 
# BERT is a newly released language understanding model from Google. For more details, see : https://github.com/google-research/bert. 
# 
# Among others, you can extract word embeeddings from the model. These are *contextualised* word embeddings.
# 
# https://github.com/imgarylai/bert-embedding

# In[11]:


from bert_embedding import BertEmbedding
import numpy as np
import string, random
import re


# In[12]:


bert_embedding = BertEmbedding()


# In[13]:


test_sentence1 = ["I bought an apple"]
# test_sentence2 = ["I bought an apple iphone"]
bert_vector1 = bert_embedding(test_sentence1)
print(bert_vector1)


# In[14]:


len(bert_vector1)


# In[15]:


bert_vector1[0][0]


# In[16]:


test_sentence1_apple_vector = bert_vector1[0][1][3]
test_sentence1_apple_vector


# In[17]:


print(len(bert_vector1[0][1][0])) # I
print(len(bert_vector1[0][1][1])) # bought
print(len(bert_vector1[0][1][2])) # an
print(len(bert_vector1[0][1][3])) # apple


# In[18]:


test_sentence2 = ["I bought an apple iphone"] # Previous sentence : I bought an apple.
bert_vector2 = bert_embedding(test_sentence2)


# In[19]:


test_sentence2_apple_vector = bert_vector2[0][1][3]
test_sentence2_apple_vector


# In[20]:


similarity_apples = compute_cosine_similarity(test_sentence1_apple_vector,test_sentence2_apple_vector)

# i bought an apple
# i bought an apple iphone

print(similarity_apples)


# In[21]:


# bought

similarity_bought = compute_cosine_similarity(bert_vector1[0][1][1],bert_vector2[0][1][1])

print(similarity_bought)


# In[ ]:





# # OOV
# 
# 
# ### The BERT model we used was trained on the book corpus and the wikipedia corpus.
# 
# ### BERT uses a WordPiece model -- if the complete word is not in its vocab, resort to subword. The lowest level of fallback is character.
# 
# ### BERT vocab consists of : ~30,000 most common words and subwords in English + all characters 

# In[23]:


oov_sentence = ["I want to learn dlkfjegsdvkn"]
bert_oov = bert_embedding(oov_sentence)
print(bert_oov)


# In[24]:


bert_oov[0][1][4]


# ## Practical use case : Lets identify near-duplicate document pairs.

# In[25]:


def split2sentences(inputText):
    splits = re.split('\.|\n',inputText)  # Dot or a newline = sentence dimiliter. Sometimes, there is no space after dot for sentences due to malformed text.
    return list(filter(None, splits)) # remove all empty elements caused by repeated dimiliters


# In[26]:


def getVector(inputText, model=None):
    
    sentences = split2sentences(inputText)
    
    result = bert_embedding(sentences)  # We can pass the array of sentences all at once to get bert embeddings.
    
    
    avg_document_vector = np.zeros(shape=(len(sentences),768)) # Each BERT vector is of length 768
    count_sentences = 0
    
    
    for i in range(len(result)):
        current_result = result[i]
        
        current_sentence_tokens = current_result[0]
        current_sentence_vectors = current_result[1]
        
        # Sometimes, due to poor sentence identification, can end up with just one/few word sentences -- ignore.
        if len(current_sentence_tokens)<3 : 
            continue

        avg_current_sentence_vector = np.mean(current_sentence_vectors, axis=0) # to take the mean of each col across vectors for vectors obtained for each token.
          
        avg_document_vector[count_sentences] = avg_current_sentence_vector
        count_sentences = count_sentences + 1
        
    
    avg_document_vector = np.mean(avg_document_vector, axis=0) # to take the mean of each col across vectors for vectors obtained for each sentence.
    
    return avg_document_vector


# In[27]:


doc1 = '''In this paper we test for the existence of a long-run relationship between investment and savings (the Feldstein-Horioka puzzle) in a panel of 18 OECD countries, 1970-2007, allowing for heterogenous breaks in the coefficients. For this purpose we develop a bootstrap panel cointegration test with breaks robust to cross-section dependence. The test suggests that, even allowing for breaks in the countries where capital control regulations changed in the sample period, there is no evidence of an investment-savings long-run relationship for the panel as a whole.'''


# In[28]:


doc2 = '''In this paper we test for the existence of a long-run savings-investments relationship in 18 OECD economies over the period 1970-2007. Although individual modelling provides only very weak support to the hypothesis of a link between savings and investments, this cannot be ruled out as individual time series tests may have low power. We thus construct a new bootstrap test for panel cointegration robust to short- and long-run dependence across units. Thid test provides evidence of a long-run savings-investments relationship in about half of the OECD economies\nexamined. The elasticities are however often smaller than 1, the value expected under no capital\nmovements.'''


# In[29]:


bert_doc1 = getVector(doc1)
bert_doc2 = getVector(doc2)


# In[30]:


similarity_docs = compute_cosine_similarity(bert_doc1,bert_doc2)

print(similarity_docs)


# In[ ]:





# ## BERT in specific text domains.

# Scientific Papers : SCIBERT (https://github.com/allenai/scibert)
# 
# Biomedical Text : BioBERT (https://github.com/dmis-lab/biobert)
# 
# Clinical Text : ClinicalBERT (https://github.com/kexinhuang12345/clinicalBERT)

# In[ ]:





# # To summarise

# BERT is : 
#     
#     Contextual
#     OOV : Word Piece model.
#     Higher feature dimensions
#     Different neural architecture : (shallow vs more deep)
#     Training dataset -- nature and size
#     State of the art results 
#     Adapted to more/many general/domain specific text
# 
# 
# 
# What was not discussed in this presentation :
# 
#     BERT is more than just word embeddings -- fine tune it for downstream NLP tasks.

# In[ ]:



I saw a cat yesterday
0  1  2  3   4

     Input Feature    Label
ex1. 0 1                2
ex2. 1 2 .              3
ex3  2 3                4


      2 .           0 1



