# Ophthalmology Domain-Specific Neural Word Embeddings
This repository provides supporting code and information on research conducted by the research group of Dr. Sophia Ying Wang on the use of word embeddings trained on ophthalmology scientific literature abstracts from Pubmed. This includes: 

- Word embeddings which are specific to the domain of ophthalmology, trained on ophthalmology scientific literature abstracts from Pubmed
- A set of analogies which are domain-specific to ophthalmology, which can be used to evaluate word embeddings on an ophthalmology-related task, and related code 
- Related code to train the ophthalmology domain-specific word embeddings, either from PubMed ophthalmology abstracts or from ophthalmology notes from the electronic medical record (EMR) 
- A neural network model extending the TextCNN architecture published by [Kim et al in 2014](https://arxiv.org/abs/1408.5882) to compare the performance of these embeddings in predicting the visual prognosis of patients with low vision using ophthalmology notes from the EMR. 

## Dependencies 
- Python 3.5+, numpy (any version), tensorflow 2.x, tensorflow_datasets, beautifulsoup 4, sqlite3

## Files 
- analogies.csv - contains the ophthalmology domain-specific analogies, word1:word2::word3:word4. The last column of words are the "wrong answers" in place of word4, the correct answer. 
- emr-cbow-embeddings.py - Trains 300-dimensional word embeddings from EMR text using word2vec CBOW architecture 
- emr-unstructured-extrinsic.py - Uses EMR embeddings to train a neural network to predict the prognosis of low vision patients using their EMR ophthalmology progress notes
- extract-emr-cbow.py - Extracts CBOW word windows from the EMR notes, removing stopwords and words with frequency <5
- extract-pubmed-cbow.py - Extracts CBOW word windows from the PubMed abstracts, removing stopwords and words with frequency <5
- glove-unstructured-extrinsic.py - Uses GloVE embeddings to train a neural network to predict the prognosis of low vision patients using their EMR ophthalmology progress notes
- parse-pubmed.py - Takes a PubMed .xml file and stores the article metadata into a sqlite database 
- pubmed-cbow-embeddings.py - Trains 300-dimensional word embeddings from PubMed abstracts using word2vec CBOW architecture 
- pubmed-unstructured-extrinsic.py - Uses PubMed embeddings to train a neural network to predict the prognosis of low vision patients using their EMR ophthalmology progress notes
- run-analogies.py - Tests performance of PubMed, EMR, and GloVe embeddings on ophthalmology domain-specific analogies 
- runholdout.py - Run the low vision prediction model on a holdout set 

## PubMed Word Embeddings
- Due to the size of the files for PubMed Word Embeddings, these can be downloaded separately from here: https://sywang.people.stanford.edu/sites/g/files/sbiybj17276/f/pubmedophthalmologywordembeddings.zip
- In the downloaded zip file you will find a plain text file with one line per word embedding, as well as a convenient Python pickle file which, when opened, loads the word embeddings into a Python dictionary where the words are the keys and the embedding vectors (in numpy format) are the values. 
