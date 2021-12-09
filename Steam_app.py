#!/usr/bin/env python

#import libraries

import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None
from PIL import Image

import plotly.express as px
import plotly.graph_objs as go
import matplotlib as plt
import seaborn as sns

#Remove the warnings
import warnings
warnings.filterwarnings('ignore')

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

# Principal dataset:
steam_data = pd.read_csv("Datasets/steam_df_clean.csv")
# Genreal infomation dataset:
steam_info = pd.read_csv('Datasets/steam_info.csv',index_col=0)
# Split platforms dataset:
platforms_count = pd.read_csv('Datasets/platforms_count.csv')
# Split genre dataset:
genres_count = pd.read_csv('Datasets/genres_count.csv')
# Split categories dataset:
categories_count = pd.read_csv('Datasets/categories_count.csv')
# Split languages dataset:
languages_count = pd.read_csv('Datasets/languages_count.csv')


#steam color pallet
color_pallet = ['#1b2838','#c7d5e0','#2a475e','#66c0f4']

############################
# First Block
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
  fig2= px.bar(developer_count, y='count', x= 'developer',
                      title='Categories that Steam games are available',
                      color_discrete_sequence = ['#66c0f4'],
                    labels={'description': 'Categorie', 
                            'percent': 'Percent of Games'}).update_layout(showlegend=False, plot_bgcolor="white")
  st.plotly_chart(fig2)


############################
# Second Block
############################

st.header("**What's your favorite Steam game?**")
games_name = st.text_input("Type the name of your favorite game: ")
selected_game = steam_info[steam_info['Game Name']  == games_name]
st.table(selected_game)



##########################
#select multiple years

st.subheader('**Select multiple years**')
years_options = st.multiselect(' ',options=all_years, default=all_years)
select_years = int(years_options)

st.subheader('What is the percentage of games for the diffenrent computer systems?')
#Select platforms based on the select_year:
platforms_select = platforms_count.loc[platforms_count['year'] == select_years]

#Plot Pie Chart

##plot the figure
fig1 = px.pie(platforms_select, values='count', names = 'platform',
                      title='Platforms that Steam games are available')
st.plotly_chart(fig1)
