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
    st.plotly_chart(fig)

#Pie Plot
def pie_plot(df,column_name):
    fig = px.pie(df, values='count', names = column_name,
                      color_discrete_sequence = color_pallet).update_traces(textinfo='percent', 
                                                                            textposition='inside',
                                                                            textfont_size=12,
                                                                            marker=dict(colors=color_pallet, 
                                                                                        line=dict(color='#000000',
                                                                                                  width=1)))
    st.plotly_chart(fig)

#############################
# Introduction
#############################

st.set_page_config(page_title="Steam Analysis Dashboard ", 
                   layout='wide')
                   
st.title("Steam Analysis")
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

steam_data = steam_data.sort_values('release_year',ascending=True)


#select one year
all_years = steam_data.release_year.unique().tolist()
st.subheader('**Select the years you want to explore**')
year_options = st.selectbox('What year you want to explore', all_years)
select_year = int(year_options)

# Static plots in two columns
col1, col2 = st.beta_columns(2)

with col1:
  st.subheader('What is the percentage of games for the diffenrent computer systems?')
  #Select platforms based on the select_year:
  platforms_select = platforms_count.loc[platforms_count['year'] == select_year]
  ##plot the figure
  fig1 = px.pie(platforms_select, values='count', names = 'platform',
              title='Platforms that Steam games are available',
              color_discrete_sequence = color_pallet)
  st.plotly_chart(fig1)

with col2:
  developer_count = steam_data.loc[steam_data['release_year'] == select_year]
  developer_count = developer_count.developer.value_counts().reset_index()
  developer_count = developer_count.rename(columns={'index':'developer', 'developer':'count'}).sort_values('count'
                                                                                                           ,ascending=False).head(15)
  #plot
  bar_plot(developer_count,'developer','Game Developers Companies','#c7d5e0')
 
############################
# Second Block
############################

st.header("**What's your favorite Steam game?**")
game_name = st.text_input("Type the name of your favorite game: ")
selected_game = steam_info[steam_info['Game Name']  == game_name]
st.table(selected_game)

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
    five_descriptiton = df[df.name.isin(five_games.name)].reset_index()
    
    print('If you liked {} you would also like to play:'.format(df_recommend.name.iloc[0]))
    print('=========================================================================')
    print('  ')
    
    
    for i in range(len(five_descriptiton)):
        print(five_descriptiton.name.loc[i])
        print('  ')
        print(five_descriptiton.short_description.loc[i])
        print('-------------------------------------------------------------------------')

#####################################################3
st.header("**Recommendation System**")
#game_id = st.text_input("Type the game id: ")
#selected_id = steam_recommend[steam_recommend['steam_appid']  == game_id]

tfidf = TfidfVectorizer()
#fitting all the words that we have for all genres using the TF-IDF approach
tfidf.fit(steam_recommend['genre'])

sparse_matrix, user_mapper, game_mapper,user_inv_mapper, game_inv_mapper = create_sparse_matrix(steam_recommend, 'user_score')
cities_recommend = top_recommend(steam_recommend,255710,k=500)
print_description(steam_recommend, cities_recommend,tfidf)
