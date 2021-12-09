#!/usr/bin/env python

#import libraries

import plotly.express as px
import plotly.graph_objs as go
import matplotlib as plt
import seaborn as sns

#Remove the warnings
import warnings
warnings.filterwarnings('ignore')

##############################################################################################
# PAGE STYLING
##############################################################################################
st.set_page_config(page_title="Steam Analysis Dashboard ", 
                   layout='wide')
                   
st.title("Steam Analysis")
