# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:43:46 2022

@author: baskar
"""

import warnings
warnings.filterwarnings('ignore')
#Data Manipulation and Treatment
import numpy as np
import pandas as pd
from datetime import timedelta

#Statistics
from keras.models import Sequential
from keras.layers import LSTM,Dense

#df = pd.read_csv('stock_data_x.csv', sep=",")
df = pd.read_csv('stock_data.csv').dropna()
l_0 = df.iloc[0].T
df_0 = l_0.to_frame()
df_0.columns = ["Day_0"]
df_fmt_0 = df_0.reset_index().rename(columns={'index':'Symbol'})
df_f0 = df_fmt_0.iloc[1: , :]
df.head()

#print(type('Date'))
df.index = df['Date']
df = df.drop('Date',axis=1)
df.head()
#print(type('Date'))
model_train=df.iloc[:int(df.shape[0]*0.80)]
valid=df.iloc[int(df.shape[0]*0.80):]
y_pred=valid.copy()
model_pred = []
i = 1
lstm_model_new_dte=[]
for d in range(1,180):
    lstm_model_new_dte.append(df.index[-1] + timedelta(days=d))
    model_predictions_dte=pd.DataFrame(zip(lstm_model_new_dte),columns=(["Date"]))          
m = model_predictions_dte
for i in range(len(df.columns)):
    tickerSymbol = df.columns[i]
#     print(tickerSymbol)
# Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    LSTM_model_new_date = []
    model_predictions_date = []
    for j in range(1,180):
# Get the models predicted price values 
        lstm_predictions = model.predict(x_test)
        lstm_predictions = scaler.inverse_transform(predictions)
        model_predictions_date=pd.DataFrame(zip(lstm_model_new_date),columns=(["Date"])) 
        x = pd.concat([model_predictions_date,model_predictions],axis=1)         
    m = pd.merge(m,x,left_on="Date",right_on="Date")
    m.to_csv('pred_data.csv',encoding='utf-8')
month = 60
l_30 = m.iloc[month].T
df_30 = l_30.to_frame()
df_30.columns = ["Day_%s" %month]
df_30.columns
df_30_fmt = df_30.reset_index().rename(columns={'index':'Symbol'})
df_f30 = df_30_fmt.iloc[1: , :].drop('Symbol', 1)
df_0_30 = pd.concat([df_f0,df_f30],axis=1)         

l_60 = m.iloc[60].T
df_60 = l_60.to_frame()
df_60.columns = ["Day_60"]
df_60_fmt = df_60.reset_index().rename(columns={'index':'Symbol'})
df_f60 = df_60_fmt.iloc[1: , :].drop('Symbol', 1)
df_0_30_60 = pd.concat([df_0_30,df_f60,],axis=1)

l_90 = m.iloc[90].T
df_90 = l_90.to_frame()
df_90.columns = ["Day_90"]
df_90_fmt = df_90.reset_index().rename(columns={'index':'Symbol'})
df_f90 = df_90_fmt.iloc[1: , :].drop('Symbol', 1)
df_0_30_60_90 = pd.concat([df_0_30_60,df_f90,],axis=1)

l_120 = m.iloc[120].T
df_120 = l_120.to_frame()
df_120.columns = ["Day_120"]
df_120_fmt = df_120.reset_index().rename(columns={'index':'Symbol'})
df_f120 = df_120_fmt.iloc[1: , :].drop('Symbol', 1)
df_0_30_60_90_120 = pd.concat([df_0_30_60_90,df_f120,],axis=1)

l_180 = m.iloc[178].T
df_180 = l_180.to_frame()
df_180.columns = ["Day_180"]
df_180_fmt = df_180.reset_index().rename(columns={'index':'Symbol'})
df_f180 = df_180_fmt.iloc[1: , :].drop('Symbol', 1)
df_final = pd.concat([df_0_30_60_90_120,df_f180,],axis=1)

df_final["Trend_30"] = (np.where(df_final['Day_0'] < df_final['Day_30'], "Up", "Down"))
df_final["Trend_90"] = (np.where(df_final['Day_0'] < df_final['Day_90'], "Up", "Down"))
df_final["Trend_180"] = (np.where(df_final['Day_0'] < df_final['Day_180'], "Up", "Down"))

df_final["Trend_V_30"] = (df_final['Day_30'] - df_final['Day_0'])/(df_final['Day_0']) * 100
df_final["Trend_V_90"] = (df_final['Day_30'] - df_final['Day_0'])/(df_final['Day_0']) * 100
df_final["Trend_V_180"] = (df_final['Day_180'] - df_final['Day_0'])/(df_final['Day_0']) * 100
df_final.to_csv('final_data.csv',encoding='utf-8',index=False)




