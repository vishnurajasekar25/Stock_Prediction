#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 7 19:33:35 2022

@author: vishnu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end   = '2022-04-30'

st.title('Prediction of Stocks')
input_user = st.text_input('Enter your ticker', 'TSLA')
df = data.DataReader(input_user, 'yahoo', start, end)
df.head()

#Describing the data

st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#Visualization
add_sidebar = st.sidebar.selectbox('Data needed', ('Opening','Closing'))
if add_sidebar == 'Opening':
    st.subheader('Opening price vs Time chart')
    fig = plt.figure(figsize =(12,6))
    plt.plot(df.Open)
    st.pyplot(fig)

    st.subheader('Opening price vs Time chart with moving average 100')
    mov_avg_100 =df.Open.rolling(100).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(mov_avg_100,'r')
    plt.plot(df.Open,'b')
    st.pyplot(fig)

    st.subheader('Opening price vs Time chart with moving average 100 and 200')
    mov_avg_200 =df.Open.rolling(200).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(mov_avg_100,'r')
    plt.plot(mov_avg_200,'g')
    plt.plot(df.Open,'b')
    st.pyplot(fig)

#Splitiing data into training and testing

    df_train = pd.DataFrame(df['Open'][0:int(len(df)*0.60)])
    df_test = pd.DataFrame(df['Open'][int(len(df)*0.60): int(len(df))])
    print(df_train.shape)
    print(df_test.shape)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    df_train_array = scaler.fit_transform(df_train)
 
#Loading the model

    model =load_model('keras_model.h5')

#Getting the values of the past 100 days for testing

    past_100_days = df_train.tail(100)
  
#Testing section

    final_df = past_100_days.append(df_test, ignore_index = True)
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    #Prediction making
    
    y_predict = model.predict(x_test)
    
    #Scaling them up and comparing
    
    scale = scaler.scale_
    scale_factor = 1/scale[0]
    y_predict = y_predict *scale_factor
    y_test = y_test * scale_factor
    
    #Graphs
    st.subheader('Predicted vs Original')
    fig2 = plt.figure(figsize =(12,6))
    plt.plot(y_test, 'r', label = 'Original price')
    plt.plot(y_predict, 'g', label = 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    
    
    
    
if add_sidebar == 'Closing':
    st.subheader('Closing price vs Time chart')
    fig = plt.figure(figsize =(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing price vs Time chart with moving average 100')
    mov_avg_100 =df.Close.rolling(100).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(mov_avg_100,'r')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

    st.subheader('Closing price vs Time chart with moving average 100 and 200')
    mov_avg_200 =df.Close.rolling(200).mean()
    fig = plt.figure(figsize =(12,6))
    plt.plot(mov_avg_100,'r')
    plt.plot(mov_avg_200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

#Splitiing data into training and testing

    df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.60)])
    df_test = pd.DataFrame(df['Close'][int(len(df)*0.60): int(len(df))])
    print(df_train.shape)
    print(df_test.shape)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    df_train_array = scaler.fit_transform(df_train)
 
#Loading the model

    model =load_model('keras_model_close.h5')

#Getting the values of the past 100 days for testing

    past_100_days = df_train.tail(100)
  
#Testing section

    final_df = past_100_days.append(df_test, ignore_index = True)
    input_data = scaler.fit_transform(final_df)
    
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    #Prediction making
    
    y_predict = model.predict(x_test)
    
    #Scaling them up and comparing
    
    scale = scaler.scale_
    scale_factor = 1/scale[0]
    y_predict = y_predict *scale_factor
    y_test = y_test * scale_factor
    
    #Graphs
    st.subheader('Predicted vs Original')
    fig2 = plt.figure(figsize =(12,6))
    plt.plot(y_test, 'r', label = 'Original price')
    plt.plot(y_predict, 'g', label = 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)