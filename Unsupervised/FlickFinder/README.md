# FlickFinder: machine learning movie recommendation system
![alt text](./static/android-chrome-512x512.png)

## Installation
### Dependencies
Two [parquet data files](https://files.stratz.me/FlickFinder) are required to run the app:

1. A subset of the movies data frame
2. The model matrix

### Docker
A docker image of this app can also be pulled from my Docker Hub [repository](https://hub.docker.com/r/dswede43/flickfinder).

#### Example
A running instance of the app can be found [here](https://flickfinder.stratz.me).

## Introduction
In an era where the vast ocean of streaming platforms offers an overwhelming array of movies and shows,
finding the perfect cinematic gem can feel like searching for a needle in a haystack. To navigate this
cinematic labyrinth and ensure viewers are treated to a tailored movie-watching experience, I embarked
on a journey to create a personalized movie recommendation system powered by machine learning in Python.

My machine learning movie recommendation system allows the user to query and select from a list of nearly
10,000 popular movies, which then returns a list of the top 5 most similar movies based on the movies plot description.

## Dataset
The TMDB movies dataset from [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
with an *Open Data Commons (ODC) Attribution License* was used to train the model. This dataset contains
24 variables for nearly 1 million movies. The main variables of interest were the movies *title*, *budget*,
*imdb_id*, *original_language*, *overview*, and *poster_path*.

### Data preprocessing
To reduced the size and usefullness of the dataset, certain movies were removed based on some filtering
criteria. These included:
1. Any movies with a non-english *original_language* were removed.
2. Any movies with a *budget* less than 1 million $USD were removed.

After row filtering of movies, column filtering was completed to include only variables needed for the
remainder of the project. These included movies *title*, *imdb_id*, *overview*, and *poster_path*. The
use of these variables will be discussed later. Additionally, any movie rows containing a NULL or NA
value in any one of these columns were also removed from the dataset. This resulted in a data frame of
9,311 movies to build the machine learning model and create the final application.

## Machine learning model
To build the model, various text processing algorithms and unsupervised machine learning methods were used.
More specifically, movie *overviews* was the only feature variable used to build the model.

### Text processing
Text processing included the lemmatization of movie description words into their simpler and standardized
form, removal of common English stop words that provide no significant meaning to a sentence, and
Part-of-Speech (POS) tagging to mark words with a specific context (ie. nouns, verbs, adjectives,
adverbs, etc.). Movie texts were then quantified into numbers using Term Frequency-Inverse Document
Frequencies (TF-IDF), which is an adjusted measure of importance of a word to a given movie relative
to a collection of movies.

### Unsupervised model training
TF-IDF values were used to calculate the cosine similarities between all combinations of movies.
This created a 9,311 by 9,311 matrix of cosine similarity values to quantify the similarities between
each movie. This matrix is the final machine learning model.

## Application and user experience
To provide a simple and easy method to query this model, a dynamic webserver application was created
using the *streamlit* library in Python. This webserver allows users to search and choose from the list
9,311 movies. Once a movie is selected, the machine learning model matrix is queried to produce a list
of movie recommendations from the top 5 movies with most similar cosine similarity score. The movie
recommendations are then queried in the original dataset for their *poster_path* and *imdb_id*. The
*poster_path* is used to display an image of the movie poster and *imdb_id* to provide a link to the
movies IMDB webpage. Users can then click on the movie recommendation poster image and learn more about
the recommended movie.

## Limitations and future directions
The main limitation is the use of only one feature to build the machine learning model. Adding more text
features such as the movie *keywords* and *tagline* that were present in the original dataset would
provide more information to create a more accurate model with better recommendations. Additionally,
other variables such as *release_date* and *adult* would create movie recommendations with more similar
release dates and allow the recommendations to differentiate between adult and children movies.

Lastly, the speed up the processing time of a movie recommendation query, Cython could be used to speed
up the functions that were written in Python by compiling the Python code to C.
