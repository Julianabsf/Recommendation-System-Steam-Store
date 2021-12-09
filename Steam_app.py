#!/usr/bin/env python

#import libraries

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib as plt

#Remove the warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#--------------Load the datasets-----------------------
# Principal dataset:
steam_data = load_data("/home/juliana/Documents/DS_Allwomen/_Final_project_dataset/steam_df_clean.csv")
# Genreal infomation dataset:
steam_info = load_data('/home/juliana/Documents/DS_Allwomen/_Final_project_dataset/steam_general.csv')
# Split platforms dataset:
# Split genre dataset:
# Split categories dataset:
# Split languages dataset:

#--------------Creat the page name----------------------

st.set_page_config(page_title="Steam Analysis- Dashboard", layout='wide')

#---------------Set the oppen menssage -----------------

st.markdown("***'How Steam Games...?'***")
st.markdown("This app alows you to explore the steam dataset and get games recommendations based on the games you liked.")


# In[ ]:


#------------------Second Block-------------------------

# Input
st.markdown("### **What are the informations about your favorite game?**")

select_name = st.text_input("Write the name of your favorite Steam Game: ")
select_name_ = steam_info[team_info['Game Name']  == select_name]

st.subheader("That's your favorite game informations")

st.table(select_name_)

