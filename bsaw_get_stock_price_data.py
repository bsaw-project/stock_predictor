#!/usr/bin/env python
# coding: utf-8

# Importing Packages
import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from google.cloud import storage

# 12 months of prior to current date
one_year = date.today() + relativedelta(months=-12)

# GCS Bucket Connection
def gcs_bucket(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    print(f"{bucket_name} is connected !!!")

    # Select the bucket created
    if bucket.exists():
        # Write the file to GCS
        write_to_gcs(bucket)    
    else:
        bucket = client.create_bucket(bucket_name)
        write_to_gcs(bucket)

# Writing CSV File to GCS
def write_to_gcs(bucket):
    # Ticker Data
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    _ = tickers.to_csv('ticker_data.csv', encoding='utf-8', index = False)

    blob = bucket.blob('data/ticker_data.csv')
    blob.upload_from_filename(filename='ticker_data.csv')

    # Stock Data from Yahoo Finance for SP500 till todate (Slider)
    todate = date.today() 
    df_stock_data = yf.download(tickers.Symbol.to_list(),one_year,todate,auto_adjust=True,progress=False)['Close']
    
    # Remove the delisted tickers and fill NaN to zeroes for new Tickers    
    df_stock_data.dropna(axis='columns', how ='all').fillna(0)
    _ = df_stock_data.to_csv('stock_data.csv', encoding='utf-8')
    
    # Write tickers into GCS bucket    
    blob = bucket.blob('data/stock_data.csv')
    blob.upload_from_filename(filename='stock_data.csv')
    print(f"Files are uploaded in the {bucket_name} !!!")

if __name__ == '__main__':
    bucket_name = "bsaw-stock-bucket-1"            # Enter Bucket name here

    gcs_bucket(bucket_name)

