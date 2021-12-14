#!/usr/bin/env python


#import libraries

#basic libraries
import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None
from PIL import Image
import numpy as np

#EDA libraries
import plotly.express as px
import plotly.graph_objs as go
import matplotlib as plt
import seaborn as sns

#Model libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

#Remove the warnings
import warnings
warnings.filterwarnings('ignore')


#########################
#Page Configuration
########################

st.set_page_config(page_title="Steam Analysis Dashboard ", layout='wide')

#############################
# Plots Functions
#############################

#Bar Plot
def bar_plot(df,column_name,xaxis_name,color):
    fig = px.bar(df, y='count', x= column_name,
                    color_discrete_sequence = [color],
                    labels={column_name: xaxis_name, 
                            'count': 'Total Games'}).update_layout(showlegend=False,
                                                                          plot_bgcolor="white")
    st.plotly_chart(fig,height=800,width=600)

#Pie Plot
def pie_plot(df,column_name):
    fig = px.pie(df, values='count', names = column_name,
                      color_discrete_sequence = color_pallet).update_traces(textinfo='percent', 
                                                                            textposition='inside',
                                                                            textfont_size=12,
                                                                            marker=dict(colors=color_pallet, 
                                                                                        line=dict(color='#000000',
                                                                                                  width=1)))
    st.plotly_chart(fig,height=800,width=600)

##############################
# Image
##############################

image = Image.open('games_image.jpg')
st.image(image, caption='Image by Branden Skeli')

#############################
# Introduction
#############################
                   
st.title("What game should I play?")
"""
Steam is one of the biggest video game digital distribution services nowadays. Their services include: installation and automatic updating of games, digital rights management (DRM), server hosting, video streaming, social networking services, cloud storage, and in-game voice and chat functionality.
On this page, you can see the results of the exploratory data analysis (EDA) and a recommendation system created from a 'Steam Dataset". This dataset has 40.584 game entrances of games released between 1998 and November 2021. ***Enjoy!!!***
"""

############################
# Load the datasets
############################

#Principal dataset:
steam_data = pd.read_csv("Datasets/steam_df_clean.csv")
#Genreal infomation dataset:
steam_info = pd.read_csv('Datasets/steam_general.csv',index_col=0)
#Split platforms dataset:
platforms_count = pd.read_csv('Datasets/platforms_count.csv')
#Split genre dataset:
genres_count = pd.read_csv('Datasets/genres_count.csv')
#Split categories dataset:
categories_count = pd.read_csv('Datasets/categories_count.csv')
#Split languages dataset:
languages_count = pd.read_csv('Datasets/languages_count.csv')
#Model dataset:
steam_recommend = pd.read_csv('Datasets/steam_recommend.csv', index_col=0)


#steam color pallet
color_pallet = ['#1b2838','#c7d5e0','#2a475e','#66c0f4']

############################
# First Block - EDA
############################

st.header("**Steam Data Overview**")
"""
The first part is the exploratory data analysis (EDA). In this part, you can find some 
information about Steam, such as the game's price, the publishers, etc. 
You can see it for all the years of the dataset, or you can filter for the years you want.

"""

####Select multiple years
steam_data = steam_data.sort_values('release_year', ascending=False)

all_years = steam_data.release_year.unique().tolist()
#st.sidebar.subheader('**Select the years you want to explore:**')
#select_year = st.sidebar.multiselect(' ',options=all_years, default=all_years)
st.subheader('**Select the years you want to explore:**')
select_year = st.multiselect(' ',options=all_years, default=all_years)

# Static plots in two columns
col1, col2 = st.beta_columns(2)

with col1:
  st.subheader('What is the percentage of games for the different computer systems?')
  #Select platforms based on the select_year:
  platforms_select = platforms_count[platforms_count.year.isin(select_year)]
  ##plot the figure
  pie_plot(platforms_select,'platform')

with col2:
    st.subheader("What are the game's user scores?")
    games_by_rating = steam_data.loc[steam_data.release_year.isin(select_year)]
    games_by_rating = games_by_rating.sort_values('user_score', ascending=False)
    #Histogram
    fig1 = px.histogram(games_by_rating, x="user_score",nbins=30,
                        color_discrete_sequence=['#c7d5e0']).update_layout(showlegend=False,plot_bgcolor="white")
    st.plotly_chart(fig1)
 
with col1:
    st.subheader('Are there free games on Steam?')
    free_games = steam_data[steam_data.release_year.isin(select_year)]
    free_games = free_games.is_free.value_counts().reset_index()
    free_games = free_games.rename(columns={'index':'free', 'is_free':'count'}).replace({False: 'Pay Games',
                                                                                         True: ' Free Games'})
    #plot
    pie_plot(free_games,'free')

with col2:
    st.subheader('How is the price distributed?')
    games_price = steam_data[steam_data.release_year.isin(select_year)]
    fig2 = px.scatter(steam_data, x="final_eur", y='user_score', size='user_score',hover_data=['name'],
                           color_discrete_sequence= ['#66c0f4'],
                           labels={'user_score': 'Game Rating', 
                                   'final_eur': 'Game Price'},
                           title = 'Steam games by Rating and Price')
    st.plotly_chart(fig2)
    
with col2:
    st.subheader('What are the supported languages?')
    languages_select = languages_count[languages_count.year.isin(select_year)].sort_values('count',ascending=False)
    #plot
    bar_plot(languages_select,'language','Supported Languages','#66c0f4')
    
with col1:
   st.subheader('What are the most common categories?')
   categories_select = categories_count[categories_count.year.isin(select_year)]
   categories_select = categories_select.sort_values('count',ascending=False)
   #plot
   bar_plot(categories_select,'category','Game Categories','#2a475e')
    
with col2:
    st.subheader('What are the games with the biggest average playtime?')
    game_playtime = steam_data[steam_data.release_year.isin(select_year)]
    game_playtime = game_playtime.sort_values('average_forever', ascending=False).head(10)
    #plot
    fig_playtime = px.bar(game_playtime, x='name',y='average_forever',
                      color_discrete_sequence = ['#c7d5e0'],
                     labels={'average_forever': 'Average playtime in minutes', 'name': 'Game Name'},
                     title = 'Games with the biggest playtime in Steam').update_layout(showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig_playtime)    

with col1:
  st.subheader('What are the most common genres?')
  genres_select = genres_count[genres_count.year.isin(select_year)].sort_values('count',ascending=False)
  #plot 
  bar_plot(genres_count,'genre','Game Genres','#2a475e')

with col2:
  st.subheader("Who are the game's developers?")
  developer_count = steam_data[steam_data.release_year.isin(select_year)]
  developer_count = developer_count.developer.value_counts().reset_index()
  developer_count = developer_count.rename(columns={'index':'developer', 'developer':'count'}).sort_values('count'
                                                                                                           ,ascending=False).head(15)
  #plot
  bar_plot(developer_count,'developer','Game Developers Companies','#c7d5e0')

with col1:
    st.subheader('Who are the games publishers?')
    publisher_count = steam_data[steam_data.release_year.isin(select_year)]
    publisher_count = publisher_count.publisher.value_counts().reset_index()
    publisher_count = publisher_count.rename(columns={'index':'publisher','publisher':'count'}).sort_values('count'
                                                                                                            ,ascending=False).head(15)
 
    #plot
    bar_plot(publisher_count,'publisher','Game Publishers Companies','#66c0f4')
 

############################
# Second Block
############################
st.header("**What's your favorite Steam game?**")

st.header("**Type the name of your favorite Steam Game?**")
#st.sidebar.header("**Type the name of your favorite Steam Game?**")
game_names = steam_info['Game Name'].unique().tolist()
games_options = st.selectbox('', game_names)
#games_options = st.sidebar.selectbox('', game_names)
selected_game = steam_info[steam_info['Game Name']  == games_options]
selected_game
#st.table(games_options)



############################
# Third Block - Recommendation System
############################

### Model Functions

def create_sparse_matrix(df, column):
    """  
    
    What the fuction is doing:
    
    Creat the sparse matrix from the role dataset based on a specific column of it.
    
    Parameters:
    
    df: the name of the data frame
    column: the name of the 'target' column in the data frame
    
    Returns:
    
    sparse_matrix: sparse matrix
    user_mapper: dict that maps user id's to user indices
    user_inv_mapper: dict that maps user indices to user id's
    game_mapper: dict that maps games id's to game indices
    game_inv_mapper: dict that maps game indices to game id's
    
    """
    
    #cheack user and games unique values:
    u_id = df['id'].nunique()
    g = df['steam_appid'].nunique()
    
    #creat a dictionary to map the ids and the games:
    user_mapper = dict(zip(np.unique(df["id"]), list(range(u_id))))
    game_mapper = dict(zip(np.unique(df["steam_appid"]), list(range(g))))
    
    user_inv_mapper = dict(zip(list(range(u_id)), np.unique(df["id"])))
    game_inv_mapper = dict(zip(list(range(g)), np.unique(df["steam_appid"])))
    
    #convert the ids into indexs :
    user_index = [user_mapper[i] for i in df['id']]
    game_index = [game_mapper[i] for i in df['steam_appid']]
    
    #creat the sparse matrix using a column of the data frame:
    sparse_matrix = csr_matrix((df[column], (game_index, user_index)), shape=(g, u_id))
    
    return sparse_matrix, user_mapper, game_mapper, user_inv_mapper, game_inv_mapper



def recommended_games(df,game_id, sparse_matrix, k,metric='cosine'):
    
    """
    What the fuction is doing:
    
    Find the neighbours from a game, based on the sparse matrix 
    criteria set before.
    
    Parameters:
    
    df: the name of the data frame
    game_id:the steam_appid of the game you're looking
    sparse_matrix: the sparse matrix created before
    k: the number of recommendations show
    metric: the metric to find the neighbours, set as default the 'cosine'
    
    Returns:
    
    The games titles in the neighbourhood of the one pre-define
    
    """
    
    neighbour_ids = []
    
    game_ind = game_mapper[game_id]
    game_to_assess = sparse_matrix[game_ind]
    
    # use KNN to find the games the are closed to the choosed:
    kNN = NearestNeighbors(n_neighbors=k, metric=metric)
    kNN.fit(sparse_matrix)

    neighbour = kNN.kneighbors(game_to_assess, return_distance=False)
    
    # map each neighbour id with the right movie_id:
    for i in range(1,k):
        n = neighbour.item(i) 
        neighbour_ids.append(game_inv_mapper[n])    
    
    #get the name of the recommended games:
    game_titles = dict(zip(df['steam_appid'], df['name']))
    game_title = game_titles[game_id]
    
    #append the titles into a list:
    neighbours_title = []
    for i in neighbour_ids:
        neighbours_title.append(game_titles[i])
        
    return neighbours_title



def top_recommend(df,game_id,k):
    
    """  
    
    What the fuction is doing:
    
    Run the function to get the top recomendations, add to the title of the games
    other information, such as genre, saved in the first data frame, then concat
    the informations of the 'target' game.
    
    Parameters:
    
    game_id:the steam_appid of the game you're looking
    k:the number of recommendations show
    df:the name of the data frame
    
    Returns:
    
    the top recommendations
    
    """
    
    #select the name of top recommendations:
    top_recommendations = recommended_games(df,game_id, sparse_matrix, k,metric='cosine')
    
    #get the genre informations from the top_recommendations saved on the general dataset:
    top_recommendations = df[df.name.isin(top_recommendations)]
    
    #add to this results the game we want to compare:
    top_recommendations = pd.concat([(df[df.steam_appid == game_id]), top_recommendations])
    
    return top_recommendations



def get_similarity_genre(df_recommend, tfidf):
    
    """
     What the fuction is doing:
     
     Compute the cosine similarity into the games genres, and save the
     five similar games in a data frame.
     
     Parameters:
     
     df_recommend: is the data frame that results of the function top_recommend(df,game_id,k)
     tfidf:is the big data frame vectorized.
     
    """
    
    #get the vectors from the suggested games:
    sparse_matrix = tfidf.transform(df_recommend['genre'])
    
    #re-arrange everything into a df
    doc_term_matrix = sparse_matrix.todense() #Return a dense matrix representation of this matrix.
    df = pd.DataFrame(doc_term_matrix, index= df_recommend.name,
                  columns=tfidf.get_feature_names())
    
    #computte the cosine between each pair of games:
    similarity = cosine_similarity(df, df) 
    
    #re-arrange the results into a df:
    sim_df = pd.DataFrame(similarity,columns = df_recommend.name, index = df_recommend.name) 
    similar = sim_df.iloc[0].sort_values(ascending= False).reset_index()
    five_recommendations = similar.loc[1:11].head(5)
    
    return five_recommendations



def print_description(df, df_recommend,tfidf):
    
    """
    What the fuction is doing:
    
    Print the name and the description of the top five games recommmended
    
    Parameters:
    df: the biggest data frame, that contains the role information
    df_recommend: data frame origin in the function top_recommend(df,game_id,k)
    
    """
    
    five_games = get_similarity_genre(df_recommend, tfidf).reset_index()
    five_recommend = df[df.name.isin(five_games.name)].reset_index()
    five_recommend = five_recommend.drop(['index','id'],axis=1).rename(columns={'name':'Game Name', 
                                                                                'steam_appid': 'Game Id',
                                                                                'genre':'Genre', 
                                                                                'final_eur':'Price', 
                                                                                'user_score':'Game Rating',
                                                                                'short_description':'Game Description'})
    
    return five_recommend

#####################################################


tfidf = TfidfVectorizer()
#fitting all the words that we have for all genres using the TF-IDF approach
tfidf.fit(steam_recommend['genre'])

st.header("**What're the recommended games?**")

st.subheader("**For each game you want a recommendation?**")
games_list = steam_recommend['name'].unique().tolist()
game_name = st.selectbox('', games_list)
selected_id = steam_recommend.loc[steam_recommend['name']  == game_name].steam_appid.unique()[0]

sparse_matrix, user_mapper, game_mapper,user_inv_mapper, game_inv_mapper = create_sparse_matrix(steam_recommend, 'user_score')
recommendations = top_recommend(steam_recommend,selected_id,k=500)
genre_recommendations = print_description(steam_recommend, recommendations,tfidf)
st.table(genre_recommendations)

###########################
# About the Author
###########################
