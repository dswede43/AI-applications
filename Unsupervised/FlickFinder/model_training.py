#!/usr/bin/python3

#Model training: Term Frequency-Inverse Document Frequency (TF-IDF)
#---
#Script to the train the model using TD-IDF and cosine similarity
#for the movie recommondation system.

#import packages
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Data preparation
#---
df = pd.read_csv("data/TMDB_movie_dataset_v11.csv.gz", compression = 'gzip')

#filter the dataset
df_filtered = df[df['original_language'] == 'en']
df_filtered = df_filtered[df_filtered['budget'] > 1000000]
df_filtered = df_filtered[['imdb_id','title','poster_path','overview']]
df_filtered = df_filtered.dropna()

#define the movies data frame
movies_df = df_filtered.drop('overview', axis = 1)
movies_df = movies_df.reset_index(drop = True)

#save the data as parquet file
movies_df.to_parquet("data/movies.parquet")

#define the data frame for ML training
movies = pd.DataFrame(df_filtered['overview'])


#Word processing and lemmatization
#---  
#define lemmatizer
lemmatizer = WordNetLemmatizer()

#retrive English stop words
stop_words = set(stopwords.words('english'))

#define POS verb codes
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

#define function to preprocess movie overview text; function was taken from:
#(https://www.geeksforgeeks.org/movie-recommender-based-on-plot-summary-using-tf-idf-vectorization-and-cosine-similarity/)
def preprocess_sentences(text):
    text = text.lower()
    temp_sent =[]
    
    #tokenize the words
    words = nltk.word_tokenize(text)
    
    #add POS tags to words
    tags = nltk.pos_tag(words)
    
    #for each word
    for i, word in enumerate(words):
        
        #if POS tag is a verb
        if tags[i][1] in VERB_CODES:
            
            #lemmatize the word with verb POS tag
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            
            #lemmatize the word
            lemmatized = lemmatizer.lemmatize(word)
            
        #remove stop words non-alphabetic characters
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)
            
    #apply final word processing
    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent

#apply preprocessing to movie overviews
movies['processed_overview'] = movies['overview'].apply(preprocess_sentences)


#TF-IDF and cosine similarity
#---
#vectorize processed movie overviews using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_mat = tfidf_vectorizer.fit_transform(movies['processed_overview'])

#calculate the cosine similarity between TF-IDF values
cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)

#convert to data frame
cosine_df = pd.DataFrame(cosine_sim)

#save model as parquet file
cosine_df.to_parquet("data/IF-IDF_cosine_sim_model.parquet")

