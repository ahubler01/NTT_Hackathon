import pandas as pd
import streamlit as st
import numpy as np
import datetime
from plotly import express as px
# for an interactive dashboard with Plotly


WORKDIR = "/home/velocitatem/Documents/Projects/JupyterNotebooks/University/Third Year/AIHackathon/notebooks/"

df_omie_labelled = pd.read_csv(WORKDIR+'../data/df_omie_labelled.csv')
df_omie_labelled.columns = ['code','description', 'datetime', 'price', 'energy']
df_omie_labelled['datetime'] = pd.to_datetime(df_omie_labelled['datetime'])
DOWNSAMPLE = True
if DOWNSAMPLE:
    df_omie_labelled = df_omie_labelled.sample(frac=0.1, random_state=42)


df_filtered_cat = pd.read_csv(WORKDIR+'../data/filtered_categories.csv')


df_unit_list = pd.read_csv(WORKDIR+'../data/unit_list.csv')
df_blind = pd.read_csv(WORKDIR+'../data/df_omie_blind.csv')
df_omie_labelled.columns = ['code','description', 'datetime', 'price', 'energy']

"""
This data is not great to work with but with a dashboard we can make it more interactive and easier to understand.
Things to consider:
+ Be able to look at each plants contribution, plot independently for each code
+ Look at the price and energy consumption for each plant over time
+ Look at total breakdown of energy consumption and price over time
"""


st.title('OMIE Energy Consumption Dashboard')

st.write('This dashboard allows you to explore the OMIE energy consumption data.')
# uers has to be able to select a plant code or more than one plant code and a time range
# for which to view the data

# Plant code selection
plant_codes = df_omie_labelled['code'].unique()
selected_codes = st.multiselect('Select plant codes', plant_codes)

# Date range selection
min_date = df_omie_labelled['datetime'].min()
max_date = df_omie_labelled['datetime'].max()
start_date = st.date_input('Start date', min_date)
start_date = pd.to_datetime(start_date)
end_date = st.date_input('End date', max_date)
end_date = pd.to_datetime(end_date)

# Filter data
mask = (df_omie_labelled['datetime'] >= start_date) & (df_omie_labelled['datetime'] <= end_date)
if selected_codes:
    mask = mask & df_omie_labelled['code'].isin(selected_codes)
df_filtered = df_omie_labelled[mask]

# Plot data
st.write('## Energy consumption and price over time')
for code, group in df_filtered.groupby('code'):
    st.write(f'### Plant code: {code}')
    st.line_chart(group.set_index('datetime')[['energy', 'price']])
st.write('## Total energy consumption and price over time')
total_group = df_filtered.groupby('datetime').sum()
st.line_chart(total_group[['energy', 'price']])


# make one chart on which you plot all the data for the selected plant codes
# but only plant data points that happen within the same minute
same_time = df_omie_labelled.groupby(['datetime', 'code']).sum().reset_index() # sum energy and price for each minute
mask = (same_time['datetime'] >= start_date) & (same_time['datetime'] <= end_date)
if selected_codes:
    mask = mask & same_time['code'].isin(selected_codes)
same_time = same_time[mask]
st.write('## Energy consumption and price for selected plant codes at the same time')
fig = px.line(same_time, x='datetime', y='energy', color='code', title='Energy consumption for selected plant codes at the same time')
st.plotly_chart(fig)
