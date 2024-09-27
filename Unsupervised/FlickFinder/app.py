#Movie recommendations system web app (StreamLit)
#---
#Script to the run the movie recommendations system web app via StreamLit

#import packages
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import helpers


#Set page customizations
#---
im = Image.open("static/favicon-16x16.png")
st.set_page_config(
    page_title = "FlickFinder",
    page_icon = im)


#Load data
#---
#load the TF-IDF cosine similarity model
cosine_sim = pd.read_parquet("data/IF-IDF_cosine_sim_model.parquet")

#load the movies data frame
movies = pd.read_parquet("data/movies.parquet")

#define the dictionary of movies
titles = dict(zip(movies.index, movies['title']))


#Streamlit front-end
#---
#set the web app title
st.title('FlickFinder')

#create select box to select movie title
title = st.selectbox('Movie recommendations similar to:', titles.values())
movie_id = helpers.get_key(titles, title)

#create button to start movie recommendations
if st.button('Recommend'):
    
    #find the recommended movies
    names, posters, IMDB_urls = helpers.recommendations(movie_id)
    
    #display the queried movie poster
    c = st.container()
    with c:
        st.write(names[0])
        st.markdown(f'<a href="{IMDB_urls[0]}"><img src="{posters[0]}" width="200"></a>', unsafe_allow_html=True)
    
    #add header
    st.markdown("<h3 style='font-size:18px;'>Recommendations</h3>", unsafe_allow_html = True)
    
    #print the recommended movie names and posters
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[1])
        st.markdown(f'<a href="{IMDB_urls[1]}"><img src="{posters[1]}" width="140"></a>', unsafe_allow_html=True)
    with col2:
        st.text(names[2])
        st.markdown(f'<a href="{IMDB_urls[2]}"><img src="{posters[2]}" width="140"></a>', unsafe_allow_html=True)
    with col3:
        st.text(names[3])
        st.markdown(f'<a href="{IMDB_urls[3]}"><img src="{posters[3]}" width="140"></a>', unsafe_allow_html=True)
    with col4:
        st.text(names[4])
        st.markdown(f'<a href="{IMDB_urls[4]}"><img src="{posters[4]}" width="140"></a>', unsafe_allow_html=True)
    with col5:
        st.text(names[5])
        st.markdown(f'<a href="{IMDB_urls[5]}"><img src="{posters[5]}" width="140"></a>', unsafe_allow_html=True)

