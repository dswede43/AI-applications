#Helper functions for streamlit app
#---

#import packages
import pandas as pd
import numpy as np


#Load data
#---
#load the TF-IDF cosine similarity model
cosine_sim = pd.read_parquet("data/IF-IDF_cosine_sim_model.parquet")

#load the movies data frame
movies = pd.read_parquet("data/movies.parquet")

#define the dictionary of movies
titles = dict(zip(movies.index, movies['title']))


#Function to fetch movie poster images
#---
def fetch_poster(movie_id, movies = movies):
    
    #fetch the poster path
    poster_path = movies[movies.index == movie_id]['poster_path'].to_string(index = False)
    return "https://image.tmdb.org/t/p/w500" + poster_path


#Function to fetch the movie IMDB id
#---
def fetch_IMDB(movie_id, movies = movies):
    
    #fetch the poster path
    IMDB_id = movies[movies.index == movie_id]['imdb_id'].to_string(index = False)
    return "https://imdb.com/title/" + IMDB_id


#Function to query the model and recommend top most similar movies
#---
def recommendations(movie_id, titles = titles, cosine_sim = cosine_sim, num_movies = 5):
    
    #recommended movie names
    names = []
    
    #recommended movie posters
    posters = []
    
    #recommended movie IMDB urls
    IMDB_urls = []
    
    #obtain and sort similarity scores for this title   
    similarity_scores = pd.Series(cosine_sim.loc[movie_id]).sort_values(ascending = False)
    
    #obtain the indices top 10 most similar movies
    top_movies = list(similarity_scores.iloc[0:num_movies + 1].index)
    
    #obtain the movie titles for each indexed movie
    for i in top_movies:
        names.append(titles.get(i))
        posters.append(fetch_poster(i))
        IMDB_urls.append(fetch_IMDB(i))
    return names, posters, IMDB_urls


#Function to extract the key from a dictionary value
#---
def get_key(my_dict, val):
    
    for key, value in my_dict.items():
        if val == value:
            return key
            
    return "key doesn't exist"

