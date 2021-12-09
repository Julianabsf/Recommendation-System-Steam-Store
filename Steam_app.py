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
steam_info = pd.read_csv('steam_general.csv')
# Split platforms dataset:
#platforms_count = pd.read_csv('/Datasets/platforms_count.csv')
# Split genre dataset:
#genres_count = pd.read_csv('/Datasets/genres_count.csv')
# Split categories dataset:
#categories_count = pd.read_csv('/Datasets/categories_count.csv')
# Split languages dataset:
#languages_count = pd.read_csv('/Datasets/languages_count.csv')

############################
# First Block
############################

st.header("**Steam Data Overview**")

