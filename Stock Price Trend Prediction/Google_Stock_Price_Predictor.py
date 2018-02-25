# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:30:20 2018

@author: Sujay
"""

# Data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60 timmesteps and output
X_Train = []
y_train = []

for i in range(60, 1258):
    X_Train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
    
X_Train, y_train =  np.array(X_Train), np.array( y_train)
    

# Reshaping (adding another dimension)
X_Train = np.reshape(X_Train, (X_Train.shape[0],X_Train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialize RNN
regressor = Sequential()

# Adding first layer of LSTM and adding dropout to prevent overfitting
regressor.add(LSTM(units=50, return_sequences= True, input_shape=(X_Train.shape[1],1)))
regressor.add(Dropout(rate=0.2))

# adding second lstm layers
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(rate=0.2))

# adding third lstm layers
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(rate=0.2))

# adding fourth lstm layers
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# adding the output layer
regressor.add(Dense(units=1))

#compiling the rnn
regressor.compile(optimizer='adam',loss='mean_squared_error')

# fitting the rnn
regressor.fit(x=X_Train,y=y_train, epochs=100, batch_size=32)

#Part 3
# Get the real google stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Get the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs =  inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
    
X_test =  np.array(X_test)

# Reshaping (adding another dimension)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#predicting and visualizing
plt.plot(real_stock_price, color='red',label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue',label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()