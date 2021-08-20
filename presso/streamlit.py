from altair.vegalite.v4.schema.core import Sort
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
# import seaborn as sns
# import matplotlib.pyplot as plt

# frame_a = pd.read_csv('C:/Users/pc/Desktop/my_git/presso/data/Sale_Q1_2021_(6).csv')
# frame_b= pd.read_csv('C:/Users/pc/Desktop/my_git/presso/data/Sale_Q2_2021.csv')
# df = frame.copy()

# Prepare the path to load dataset
path = 'C:/Users/pc/Desktop/my_git/presso/data/'
list_data = ['Sale_Q1_2021_(6).csv','Sale_Q2_2021.csv']
data_path = []
for i in list_data:
    dt_path = path + i
    data_path.append(dt_path)

frame = []
for j in data_path:
    frame_x = pd.read_csv(j)
    frame.append(frame_x)

# Read the dataset
df = pd.concat(frame,axis=0, ignore_index=True)

st.title('Sales Dashboard')
st.write('Summary')

st.selectbox('Select year', [2021,2020])
# Sum of revenue group by Date & Product:
df_revenue = df.groupby(['date','product_name'])[['total_cost','revenue','profit','quantity']].sum().reset_index()
df_revenue_monthly = df_revenue.groupby('date')[['profit','quantity']].sum().reset_index()
# Chart:
chart = alt.Chart(df_revenue_monthly).mark_bar(color='#96ceb4').encode(# column='product_name',
                                                        x='date',
                                                        y='profit').properties(width=400) 
st.write(chart)

# Sales by Drink Category :
st.write('Sales by Drink Category')
drink = st.multiselect('Please select:', [i for i in df_revenue['product_name'].unique()])
if len(drink) != 0:
    df_drink = df_revenue[df_revenue['product_name']==drink[0]]
    df_drink
    df_melt = df_drink.melt(id_vars=['date'], value_vars=['total_cost','revenue','profit'])
    # chart = alt.Chart(df_drink).mark_bar().encode(column='product_name',
    #                                                 x=alt.X('date',sort=None),
    #                                                 y='profit')
    chart = alt.Chart(df_melt).mark_bar().encode(column='variable',
                                                    x=alt.X('date',sort=None),
                                                    y='sum(value)',
                                                    color='date').properties(width=200) 
    st.write(chart)

# Sales by Time period:
st.write('Sales by Time period')
opt = st.multiselect('Please select:', ['Year', 'Month'])
if len(opt) != 0:
    if opt[0] == 'Year':
        opt_year = st.selectbox('Select',[2020,2021])
        if opt_year == 2021:
            df_revenue
    elif opt[0] == 'Month': 
        # month = st.selectbox('Select month', [f'{i}-21' for i in ['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec']])
        month = st.selectbox('Select month', [f'{i}-2021' for i in range(1,13)])
        if month != 0:
            #Sum of monthly sales:
            st.text('Monthly Sales')
            sum_revenue = df_revenue_monthly[df_revenue_monthly['date']==month].sum()
            sum_revenue

            # Detail of monthly sales:
            sort_revenue = df_revenue[df_revenue['date']==month].sort_values(['profit'], ascending=False)
            sort_revenue

            # Chart of monthly sales:
            df_month = df.groupby(['date','product_name'])[['total_cost','revenue','profit']].sum().reset_index()
            df_month_value = df_month[df_month['date']==month]
            df_melt = df_month_value.melt(id_vars=['product_name'], value_vars=['total_cost','revenue','profit'])
            
            all_value= alt.Chart(df_melt).mark_bar().encode(x=alt.X('product_name', sort='-y'),
                                                            y='sum(value)', 
                                                            color='variable').properties(height=500, width=800)                                    
            st.write(all_value)

            # Sort sales values:
            # chart= alt.Chart(sort_revenue).mark_bar().encode(x='profit',y=alt.Y('product_name',sort='-x'))                                              
            # st.write(chart)
            chart= alt.Chart(df_melt).mark_bar().encode(column = 'variable',x='sum(value)',y=alt.Y('product_name',sort='-x'),color='variable').properties(height=700, width=350)                                               
            st.write(chart)



