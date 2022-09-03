# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:35:09 2022

@author: baskar
"""

import warnings
warnings.filterwarnings('ignore')

#Data Manipulation and Treatment
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta

#Plotting and Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Statistics
import statsmodels.api as sm
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from pmdarima import auto_arima

#Scikit-Learn for Modeling
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error,mean_squared_log_error

# GCP Storage Package
from google.cloud import storage

def get_data_gcs(bucket_name, file_name):
    path = f'gs://{bucket_name}/data/{file_name}.csv'
    data = pd.read_csv(path)[1:].dropna(axis=1)
    return data

def prediction(df):
    # Taking Current Date convert to Dataframe
    Day_0 = df.iloc[-1].T
    df_Day_0 = Day_0.to_frame()
    df_Day_0.columns = ["Day_0"]
    df_fmt_Day_0 = df_Day_0.reset_index().rename(columns={'index':'Ticker'})
    df_f0 = df_fmt_Day_0.iloc[1: , :]
    #df.head()

    df['Date'] =  pd.to_datetime(df.Date,format='%')
    #print(type('Date'))

    df.index = df['Date']
    df = df.drop('Date',axis=1)
    df.head()
    #print(type('Date'))
    # Train date 80% and Validate Data 20%
    model_train=df.iloc[:int(df.shape[0]*0.80)]
    valid=df.iloc[int(df.shape[0]*0.80):]

    # Initialize the Model scores for auto.ARIMA model

    y_pred=valid.copy()
    model_scores_r2=[]
    model_scores_mse=[]
    model_scores_rmse=[]
    model_scores_mae=[]
    model_scores_rmsle=[]
    pred_arima=[]
    model_pred = []
    # auto.ARIMA model Training and Prediction for 180 days
    i = 1
    ARIMA_model_new_dte=[]
    for d in range(1,180):
        ARIMA_model_new_dte.append(df.index[-1] + timedelta(days=d))
        model_predictions_dte=pd.DataFrame(zip(ARIMA_model_new_dte),columns=(["Date"]))          
    m = model_predictions_dte
    for i in range(len(df.columns)):
        tickerSymbol = df.columns[i]
        # print(tickerSymbol)
        model_arima = auto_arima(model_train[tickerSymbol],trace=False, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,
                suppress_warnings=True,stepwise=False,seasonal=False)
        model_arima.fit(model_train[tickerSymbol])
        prediction_arima = model_arima.predict(len(valid))
        y_pred["ARIMA Model Prediction"] = prediction_arima
		# Model Scores
        r2_arima= r2_score(y_pred[tickerSymbol],y_pred["ARIMA Model Prediction"])
        mse_arima= mean_squared_error(y_pred[tickerSymbol],y_pred["ARIMA Model Prediction"])
        rmse_arima=np.sqrt(mean_squared_error(y_pred[tickerSymbol],y_pred["ARIMA Model Prediction"]))
        mae_arima=mean_absolute_error(y_pred[tickerSymbol],y_pred["ARIMA Model Prediction"])
        rmsle_arima = np.sqrt(mean_squared_log_error(y_pred[tickerSymbol],y_pred["ARIMA Model Prediction"]))
        model_scores_r2.append(r2_arima)
        model_scores_mse.append(mse_arima)
        model_scores_rmse.append(rmse_arima)
        model_scores_mae.append(mae_arima)
        model_scores_rmsle.append(rmsle_arima)

        ARIMA_model_new_date=[]
        ARIMA_model_new_prediction=[]

        for j in range(1,180):
            ARIMA_model_new_date.append(df.index[-1] + timedelta(days=j))
            ARIMA_model_new_prediction.append(model_arima.predict(len(valid)+j)[-1])
            pd.set_option('display.float_format', lambda x: '%.6f' % x)
            model_predictions=pd.DataFrame(zip(ARIMA_model_new_prediction),columns=([tickerSymbol]))   
            model_predictions_date=pd.DataFrame(zip(ARIMA_model_new_date),columns=(["Date"])) 
            x = pd.concat([model_predictions_date,model_predictions],axis=1)         
        m = pd.merge(m,x,left_on="Date",right_on="Date")
        m.to_csv('pred_data.csv',encoding='utf-8', index=False)
        
    # 30 days extraction of Prediction data Transformation (Include header)
    Day_30 = m.iloc[31].T
    df_Day_30 = Day_30.to_frame()
    df_Day_30.columns = ["Day_30"]
    df_Day_30_fmt = df_Day_30.reset_index().rename(columns={'index':'Symbol'})
    df_f30 = df_Day_30_fmt.iloc[1: , :].drop('Symbol', 1)
    df_Day_0_30 = pd.concat([df_f0,df_f30],axis=1)         

    # 60 days extraction of Prediction data Transformation (Include header)
    Day_60 = m.iloc[61].T
    df_Day_60 = Day_60.to_frame()
    df_Day_60.columns = ["Day_60"]
    df_Day_60_fmt = df_Day_60.reset_index().rename(columns={'index':'Symbol'})
    df_f60 = df_Day_60_fmt.iloc[1: , :].drop('Symbol', 1)
    df_Day_0_30_60 = pd.concat([df_Day_0_30,df_f60,],axis=1)

    # 90 days extraction of Prediction data Transformation (Include header)
    Day_90 = m.iloc[91].T
    df_Day_90 = Day_90.to_frame()
    df_Day_90.columns = ["Day_90"]
    df_Day_90_fmt = df_Day_90.reset_index().rename(columns={'index':'Symbol'})
    df_f90 = df_Day_90_fmt.iloc[1: , :].drop('Symbol', 1)
    df_Day_0_30_60_90 = pd.concat([df_Day_0_30_60,df_f90,],axis=1)

    # 120 days extraction of Prediction data Transformation (Include header)
    Day_120 = m.iloc[121].T
    df_Day_120 = Day_120.to_frame()
    df_Day_120.columns = ["Day_120"]
    df_Day_120_fmt = df_Day_120.reset_index().rename(columns={'index':'Symbol'})
    df_f120 = df_Day_120_fmt.iloc[1: , :].drop('Symbol', 1)
    df_Day_0_30_60_90_120 = pd.concat([df_Day_0_30_60_90,df_f120,],axis=1)

    # 180 days extraction of Prediction data Transformation (Include header)
    Day_180 = m.iloc[178].T
    df_180 = Day_180.to_frame()
    df_180.columns = ["Day_180"]
    df_180_fmt = df_180.reset_index().rename(columns={'index':'Symbol'})
    df_f180 = df_180_fmt.iloc[1: , :].drop('Symbol', 1)
    df_final = pd.concat([df_Day_0_30_60_90_120,df_f180,],axis=1)

    # Setting the Trend based on Prediction for 30 , 60 ,90 , 120 and 180 days
    df_final["Trend_30"] = (np.where(df_final['Day_0'] < df_final['Day_30'], "Up", "Down"))
    df_final["Trend_90"] = (np.where(df_final['Day_0'] < df_final['Day_90'], "Up", "Down"))
    df_final["Trend_180"] = (np.where(df_final['Day_0'] < df_final['Day_180'], "Up", "Down"))

    df_final["Trend_V_30"] = (df_final['Day_30'] - df_final['Day_0'])/(df_final['Day_0']) * 100
    df_final["Trend_V_90"] = (df_final['Day_30'] - df_final['Day_0'])/(df_final['Day_0']) * 100
    df_final["Trend_V_180"] = (df_final['Day_180'] - df_final['Day_0'])/(df_final['Day_0']) * 100

    Trend_ranking = []
    for i in df_final['Trend_30']:
        if i == "Down":
            Trend_ranking.append(2)
        elif i == "Up":
            Trend_ranking.append(1)
    df_final['Trend_rank'] = Trend_ranking

    volatile_rank = []
    for i in df_final['Trend_V_180']:
        if i > 50:
            volatile_rank.append(1)
        elif i > 25 and i <= 50:
            volatile_rank.append(2)
        elif i <= 25:
            volatile_rank.append(3)
    df_final['Volatile'] = volatile_rank
    df_final.to_csv('final_data.csv',encoding='utf-8',index=False)

# Writing to GCS 
def write_to_gcs(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob('data/pred_data.csv')
    blob.upload_from_filename(filename='pred_data.csv')
    print("pred_data uploaded !!!")

    # Final Data
    blob = bucket.blob('data/final_data.csv')
    blob.upload_from_filename(filename='final_data.csv')
    print("final data uploaded !!!")

if __name__ == '__main__':
    bucket_name = "bsaw-stock-bucket-1"
    file_name = "stock_data"
    df = get_data_gcs(bucket_name, file_name)
    prediction(df)
    write_to_gcs(bucket_name)