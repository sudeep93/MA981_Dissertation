import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy import sparse
from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '40c05f4085c354479dc351c824a4b505'

from tmdbv3api import Movie

# load the nlp model and tfidf vectorizer from disk
filename = 'a.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('b.pkl','rb'))

def similarity_tags():
    movie_df_ex = pd.read_csv('tags_recom.csv')
#     movie_df_ex=movie_df_ex.drop(['level_0', 'index'], axis=1)
    movie_df_ex['title']=movie_df_ex['title'].str.lower()
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    c  = CountVectorizer()
    tfidf_mtx = c.fit_transform(movie_df_ex['description'].astype('U').values)
    cosine_sim = linear_kernel(tfidf_mtx, tfidf_mtx)
    movie_df_ex = movie_df_ex.reset_index()
    titles = movie_df_ex['title']
    indices = pd.Series(movie_df_ex.index, index=movie_df_ex['title'])
    return movie_df_ex,indices,cosine_sim,titles


def recommendations_tags(title):
    title = title.lower()
    movie_df_ex,indices,cosine_sim,titles = similarity_tags()
#     print(movie_df_ex['title'].lower())
    if title not in movie_df_ex['title'].str.lower().unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        # print(sim_scores)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        i = titles.iloc[movie_indices]
#         print(titles.iloc[movie_indices])
        l = []
        for i in titles.iloc[movie_indices]:
            l.append(i)
        print(l)
    return l

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+434.0) * R) + (434.0/(434.0+v) * 5.244896612406511)

def similarity_metadata():
    smovie_df = pd.read_csv('content_recom.csv')
    smovie_df['title']=smovie_df['title'].str.lower()
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smovie_df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    smovie_df = smovie_df.reset_index()
    titles = smovie_df['title']
    indices = pd.Series(smovie_df.index, index=smovie_df['title'])
    return smovie_df,indices,cosine_sim,titles


# def get_similar(movie_name,rating,corrMatrix):
#     similar_ratings = corrMatrix[movie_name]*(rating-2.5)
#     similar_ratings = similar_ratings.sort_values(ascending=False)
# #     print(similar_ratings[1])
#     return similar_ratings

# def colab(movie_name,rating):
#     userRatings = pd.read_csv('ratings.csv')
#     corrMatrix = userRatings.corr(method='pearson')
# #     corrMatrix.head(100)
#     similar_movies = pd.DataFrame()
#     similar_movies = get_similar(movie_name,rating,corrMatrix).head(11) 
#     similar_movies = similar_movies.to_frame()
#     similar_movies = similar_movies.rename(index={0: "movies", 1: "sim"}).reset_index()
#     # print(similar_movies.iloc[1:11,0:1])
#     l = []
#     for i in similar_movies.iloc[1:11,0:1]:
#         l.append(i)
#     return l
def colab(movie_name,rating):
    userRatings = pd.read_csv('ratings.csv')
    corrMatrix = userRatings.corr(method='pearson')
#     corrMatrix.head(100)
    similar_movies = pd.DataFrame()
    similar_movies = corrMatrix[movie_name]*(float(rating) - 2.5)
    similar_movies = similar_movies.sort_values(ascending=False)
#     similar_movies = get_similar(movie_name,rating,corrMatrix).head(11) 
    similar_movies = similar_movies.to_frame()
    similar_movies = similar_movies.rename(index={0: "movies", 1: "sim"}).reset_index()
    # print(similar_movies.iloc[1:11,0:1])
    l = []
    for i in similar_movies['index'].iloc[1:11]:
        l.append(i)
    return l
    # return similar_movies.iloc[1:11,0:1]


def recommendations_content(title):    
    title = title.lower()
    smovie_df,indices,cosine_sim,titles = similarity_metadata()
    if title not in smovie_df['title'].str.lower().unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = smovie_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
#         print(titles.iloc[movie_indices])
#         print(qualified)
        l = []
        for i in qualified['title'].head(10):
            l.append(i)
        print(l)
    return l

def ListOfGenres(genre_json):
    if genre_json:
        genres = []
        genre_str = ", " 
        for i in range(0,len(genre_json)):
            genres.append(genre_json[i]['name'])
        return genre_str.join(genres)

def date_convert(s):
    MONTHS = ['January', 'February', 'Match', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December']
    y = s[:4]
    m = int(s[5:-3])
    d = s[8:]
    month_name = MONTHS[m-1]

    result= month_name + ' ' + d + ' '  + y
    return result

def MinsToHours(duration):
    if duration%60==0:
        return "{:.0f} hours".format(duration/60)
    else:
        return "{:.0f} hours {} minutes".format(duration/60,duration%60)

def get_suggestions():
    data = pd.read_csv('content_recom.csv')
    return list(data['title'].str.capitalize())


app = Flask(__name__)

@app.route("/")
def home():
    suggestions = get_suggestions()
    return render_template('home.html')


@app.route("/recommend",methods=['POST'])
def recommend():
    # movie = request.args.get('title' # get movie name from the URL
    movie_recom = request.form['movie']
    print("================>",movie_recom)
    types = request.form['sel']
    print("================>",types)

    print(movie_recom,types)
    if types=='tag':
        print("Taaaaaaaaag======================>")

        r = recommendations_tags(movie_recom)

        movie_recom = movie_recom.upper()
        if type(r)==type('string'): # no such movie found in the database
            return render_template('recommend.html',movie=movie_recom,r=r,t='s')
        else:
            tmdb_movie = Movie()
            result = tmdb_movie.search(movie_recom)

            # id and name of the movie
            movie_id = result[0].id
            movie_name = result[0].title
            
            # calling api
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            imdb_id = data_json['imdb_id']
            poster = data_json['poster_path']
            img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)

            
            genre = ListOfGenres(data_json['genres'])

            # web scraping reviews
            sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
            soup = bs.BeautifulSoup(sauce,'lxml')
            soup_result = soup.find_all("div",{"class":"text show-more__control"})

            reviews_list = [] 
            reviews_status = [] 
            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    # passing the review to our model
                    movie_review_list = np.array([reviews.string])
                    movie_vector = vectorizer.fit(movie_review_list)
#                     print(movie_vector.shape)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Positive' if pred else 'Negative')

            # merging reviews 
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 

            # reading votes
            votes = "{:,}".format(result[0].vote_count)
            
            # change date format
            rd = date_convert(result[0].release_date)

            # movie status
            status = data_json['status']

            # convert minutes to hours minutes (eg. 148 minutes to 2 hours 28 mins)
            runtime = MinsToHours(data_json['runtime'])

            # getting the posters for the recommended movies
            poster = []
            movie_title_list = []
            for movie_title in r:
                list_result = tmdb_movie.search(movie_title)
                movie_id = list_result[0].id
                response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
                data_json = response.json()
                poster.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
            movie_list = {poster[i]: r[i] for i in range(len(r))}

            return render_template('recommend.html',movie=movie_recom,mtitle=r,t='l',cards=movie_list,
            result=result[0],reviews=movie_reviews,img_path=img_path,genres=genre,vote_count=votes,
            release_date=rd,runtime=runtime)
    elif types=='colab':
        print("Colab======================>")
        ratings = request.form['ratings']
        print("================>",ratings)
        r = colab(movie_recom,ratings)
        print("movie recom======>",r)
        movie_recom = movie_recom.upper()
        if type(r)==type('string'): # no such movie found in the database
            return render_template('recommend.html',movie=movie_recom,r=r,t='s')
        else:
            tmdb_movie = Movie()
            result = tmdb_movie.search(movie_recom)

            # get movie id and movie title
            movie_id = result[0].id
            movie_name = result[0].title
            
            # making API call
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            imdb_id = data_json['imdb_id']
            poster = data_json['poster_path']
            img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)

            # getting list of genres form json
            genre = ListOfGenres(data_json['genres'])

            # web scraping to get user reviews from IMDB site
            sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
            soup = bs.BeautifulSoup(sauce,'lxml')
            soup_result = soup.find_all("div",{"class":"text show-more__control"})

            reviews_list = [] # list of reviews
            reviews_status = [] # list of comments (good or bad)
            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    # passing the review to our model
                    movie_review_list = np.array([reviews.string])
                    movie_vector = vectorizer.transform(movie_review_list)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Positive' if pred else 'Negative')

            # combining reviews and comments into dictionary
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 

            # getting votes with comma as thousands separators
            votes = "{:,}".format(result[0].vote_count)
            
            # convert date to readable format (eg. 10-06-2019 to June 10 2019)
            rd = date_convert(result[0].release_date)

            # getting the status of the movie (released or not)
            status = data_json['status']

            # convert minutes to hours minutes (eg. 148 minutes to 2 hours 28 mins)
            runtime = MinsToHours(data_json['runtime'])

            # getting the posters for the recommended movies
            poster = []
            movie_title_list = []
            for movie_title in r:
                list_result = tmdb_movie.search(movie_title)
                print("list_result======>",list_result)
                try:
                    movie_id = list_result[0].id
                    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
                    data_json = response.json()
                    poster.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
                except IndexError:
                    gotdata = 'null'
            movie_list = {poster[i]: r[i] for i in range(len(r))}

            return render_template('recommend.html',movie=movie_recom,mtitle=r,t='l',cards=movie_list,
            result=result[0],reviews=movie_reviews,img_path=img_path,genres=genre,vote_count=votes,
            release_date=rd,runtime=runtime)

    else :
        r = recommendations_content(movie_recom)

        movie_recom = movie_recom.upper()
        if type(r)==type('string'): # no such movie found in the database
            return render_template('recommend.html',movie=movie_recom,r=r,t='s')
        else:
            tmdb_movie = Movie()
            result = tmdb_movie.search(movie_recom)

            # get movie id and movie title
            movie_id = result[0].id
            movie_name = result[0].title
            
            # making API call
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            imdb_id = data_json['imdb_id']
            poster = data_json['poster_path']
            img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)

            # getting list of genres form json
            genre = ListOfGenres(data_json['genres'])

            # web scraping to get user reviews from IMDB site
            sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
            soup = bs.BeautifulSoup(sauce,'lxml')
            soup_result = soup.find_all("div",{"class":"text show-more__control"})

            reviews_list = [] # list of reviews
            reviews_status = [] # list of comments (good or bad)
            for reviews in soup_result:
                if reviews.string:
                    reviews_list.append(reviews.string)
                    # passing the review to our model
                    movie_review_list = np.array([reviews.string])
                    movie_vector = vectorizer.transform(movie_review_list)
                    pred = clf.predict(movie_vector)
                    reviews_status.append('Positive' if pred else 'Negative')

            # combining reviews and comments into dictionary
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 

            # getting votes with comma as thousands separators
            votes = "{:,}".format(result[0].vote_count)
            
            # convert date to readable format (eg. 10-06-2019 to June 10 2019)
            rd = date_convert(result[0].release_date)

            # getting the status of the movie (released or not)
            status = data_json['status']

            # convert minutes to hours minutes (eg. 148 minutes to 2 hours 28 mins)
            runtime = MinsToHours(data_json['runtime'])

            # getting the posters for the recommended movies
            poster = []
            movie_title_list = []
            for movie_title in r:
                list_result = tmdb_movie.search(movie_title)
                movie_id = list_result[0].id
                response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
                data_json = response.json()
                poster.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
            movie_list = {poster[i]: r[i] for i in range(len(r))}

        # # get movie names for auto completion
        # suggestions = get_suggestions()
        
        return render_template('recommend.html',movie=movie_recom,mtitle=r,t='l',cards=movie_list,
            result=result[0],reviews=movie_reviews,img_path=img_path,genres=genre,vote_count=votes,
            release_date=rd,runtime=runtime)

if __name__ == '__main__':
    app.run(debug=True)
