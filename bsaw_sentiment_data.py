import json
import logging
from pymongo import MongoClient
from pandas import DataFrame
import pandas as pd
from google.cloud import storage

import datetime
import requests

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, ArrayType
from pyspark.sql.functions import explode
from pyspark.sql.window import Window
import pyspark.sql.functions as f

client=None

def get_database():
    try:
        CONNECTION_STRING = "mongodb+srv://bsaw1234:bsaw%401234@bsaw-cluster-1.rxjnf.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
        clients = MongoClient(CONNECTION_STRING)
        tbl="stock_db"
        return clients[tbl]
    except Exception as err:
        logging.error(f'Error while opening database connection: {err}')
        return False
    
def close_connection():
    try:
        client.close()
        return True
    except Exception as err:
        logging.error(f'Error while closing database connection: {err}') #Need to add logger here
        return False
 
'''
# Cloud Function to insert the Stock into MongoDB or Trigger using CLOUD Scheduler
def insert_stockData():
    dbname = get_database()
    collection_name = dbname[config['tbl_name']]
    response_API = requests.get(config['stock_api'])
    stock_data=response_API.json()
    stock_data['created_date']=datetime.datetime.now()
    print(stock_data)
    collection_name.insert_many([stock_data])
'''

def sentiments():
    dbname = get_database()

    # Create a new collection
    collection_name = dbname["stock_info"]
    item_details = collection_name.find()

    # convert the dictionary objects to dataframe
    items_df = DataFrame(item_details)

    # Create PySpark SparkSession
    spark = SparkSession.builder.master("local[*]").appName("Convert to PySpark").getOrCreate()

    # Create Schema
    schema = StructType([
    StructField('Ticker', ArrayType(StringType(), True), True),
    StructField('Sentiment', StringType(), True),
    ])

    # Create empty DataFrame from empty RDD
    emptyRDD = spark.sparkContext.emptyRDD()
    newDF = spark.createDataFrame(emptyRDD,schema)

    # Loop over each data
    for i in range(len(items_df)):
        # Creating a DataFrame for each Object ID
        sparkDF = spark.createDataFrame(items_df['data'][i])
        
        # Appending data in the empty DataFrame
        newDF = newDF.select('Ticker', 'Sentiment').union(sparkDF.select('tickers', 'sentiment'))
        
    # Exploding the array in the 'tickers'
    df = newDF.select(explode(newDF.Ticker), newDF.Sentiment)

    # Grouping the sentiment using the 'tickers' column in descending order
    data = df.groupBy('col', 'Sentiment').count().orderBy('count', ascending=False)

    # Rename the Column Name
    data = data.withColumnRenamed("col","Ticker")

    # Count the max sentiment with order by the count
    # data = data.join(data.groupBy('Ticker').agg(f.max('count').alias('count')),on='count',how='leftsemi').orderBy('count', ascending=False)
    # Count the max sentiment with order by the count
    w = Window.partitionBy('Ticker')
    data = data.withColumn('counts', f.max('count').over(w)).where(f.col('count') == f.col('counts')).drop('counts')

    # Drop the count table (Optional)
    data = data.drop('count')

    # Add 'ss_rank' column
    data = data.withColumn('ss_rank', f.when((f.col('Sentiment') == "Positive"), 1) .when((f.col('Sentiment') == "Neutral"), 2) .when((f.col('Sentiment') == "Negative"), 3))

    # using Pandas as whole CSV
    data.toPandas().to_csv('sentiment_output.csv', index = False)

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
    blob = bucket.blob('data/sentiment_output.csv')
    blob.upload_from_filename(filename='sentiment_output.csv')
    print(f"Files are uploaded in the {bucket_name} !!!")

if __name__ == '__main__':
    sentiments()
    bucket_name = "bsaw-stock-bucket-1"            # Enter Bucket name here
    gcs_bucket(bucket_name)