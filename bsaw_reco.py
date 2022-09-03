from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import explode, col
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.cloud import storage

def ALS_model():
    bqclient = bigquery.Client()

    # Client Data File
    table = bigquery.TableReference.from_string(
        "bsaw-stock-pred-347218.Client.bsaw-client"
    )
    rows = bqclient.list_rows(table)
    client = rows.to_dataframe(create_bqstorage_client=True)

    # Portfolio Data File
    table = bigquery.TableReference.from_string(
        "bsaw-stock-pred-347218.Client.bsaw-portfolio"
    )
    rows = bqclient.list_rows(table)
    portfolio = rows.to_dataframe(create_bqstorage_client=True)

    # Download query results.
    query_string = """
    SELECT *
    FROM `bsaw-stock-pred-347218.Client.bsaw-client` AS cd
    JOIN `bsaw-stock-pred-347218.Client.bsaw-portfolio` AS pf ON cd.Client_ID = pf.Client_ID;
    """

    data = (
        bqclient.query(query_string)
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
        )
    )

    # Sentiment Output Data File
    sentiment = pd.read_csv('gs://bsaw-stock-bucket-1/data/sentiment_output.csv')

    # Stock Data File
    stock = pd.read_csv('gs://bsaw-stock-bucket-1/data/final_data.csv')

    #Create PySpark SparkSession
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Recommendation System using ALS") \
        .getOrCreate()

    #Create PySpark DataFrame from Pandas
    clientDF = spark.createDataFrame(client) 
    portfolioDF = spark.createDataFrame(portfolio) 
    sentimentDF = spark.createDataFrame(sentiment) 
    stockDF = spark.createDataFrame(stock) 
    stockDF = stockDF.select('Ticker','Day_0','Day_30','Day_90','Day_180','Trend_180', 'Trend_rank', 'volatile')

    # Select the necessary columns
    pfolioDF = portfolioDF.select('Client_Name','Ticker','Concentration')

    # Convert String Data Column to Integer Data Column (Label Encoding)
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_Index").fit(pfolioDF) for column in ['Client_Name', 'Ticker']]

    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(pfolioDF).transform(pfolioDF)
    df = df.drop(*('Client_Name', 'Ticker'))

    # Create test and train set
    (train, test) = df.randomSplit([0.8, 0.2])

    # ALS Model
    als = ALS(userCol="Client_Name_Index", itemCol="Ticker_Index", ratingCol="Concentration")
    model = als.fit(train)

    # Hyper-Parameter Tuning
    # Add hyperparameters and their respective values to param_grid
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 50, 100, 150]) \
                .addGrid(als.regParam, [.01, .05, .1, .15]) \
                .build()

    # Define evaluator as RMSE and print length of evaluator
    evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="Concentration", 
            predictionCol="prediction") 

    print ("Num models to be tested: ", len(param_grid))

    # Build cross validation using CrossValidator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    #Fit cross validator to the 'train' dataset
    model = cv.fit(train)
    best_model = model.bestModel

    # Generate n Recommendations for all users
    recommendations = best_model.recommendForAllUsers(50)

    # Expanding the Recommendations
    nrecommendations = recommendations\
        .withColumn("rec_exp", explode("recommendations"))\
        .select('Client_Name_Index', col("rec_exp.Ticker_Index"), col("rec_exp.rating"))

    recoDF = nrecommendations.withColumnRenamed('rating', 'Concentration')

    # Replacing Indexes with the Names
    # Get Metadata of the Indexed DataFrame
    meta = [
        f.metadata for f in df.schema.fields if f.name == "Client_Name_Index"
    ]

    # Convert Metadata to Dictionary
    metadata = dict(enumerate(meta[0]["ml_attr"]["vals"]))

    # Convert to List and make a DataFrame
    list_data = list(map(list, metadata.items()))
    df2 = spark.createDataFrame(list_data, ["Index", "Client_Name"])

    # Get Metadata of the Indexed DataFrame
    meta = [
        f.metadata for f in df.schema.fields if f.name == "Ticker_Index"
    ]

    # Convert Metadata to Dictionary
    metadata = dict(enumerate(meta[0]["ml_attr"]["vals"]))

    # Convert to List and make a DataFrame
    list_data = list(map(list, metadata.items()))
    df3 = spark.createDataFrame(list_data, ["Index", "Ticker"])

    # Join the DataFrames
    newDF = recoDF.join(df2, recoDF.Client_Name_Index == df2.Index, "full")
    newDF = newDF.join(df3, newDF.Ticker_Index == df3.Index, "full")
    newDF = newDF.drop(*('Client_Name_Index', 'Ticker_Index', 'Index', 'Index'))
    newDF = newDF.na.drop()

    # Appending all other DataFrames with the predicted ones
    reco_out = newDF.join(pfolioDF.select('Client_Name', 'Ticker'),['Ticker', 'Client_Name'], how="leftanti")
    reco_out = reco_out.join(stockDF, ['Ticker'])
    reco_out = reco_out.join(sentimentDF, ['Ticker'])
    reco_out = reco_out.join(clientDF, ['Client_Name'])

    # Buy/Sell/Hold
    reco_out = reco_out.withColumn('Buy/Not', 
                                f.when((f.col('Trend_rank') == 1) & 
                                        ((f.col('ss_rank') == 1) | 
                                        (f.col('ss_rank') == 2) | 
                                        (f.col('ss_rank') == 3)), "Buy")\
                                
                                .when((f.col('Trend_rank') == 2) & 
                                        ((f.col('ss_rank') == 1)),"Buy")\
                                
                                .when((f.col('Trend_rank') == 2) &
                                        ((f.col('ss_rank') == 2) | 
                                        (f.col('ss_rank') == 3)), "Don't Buy")\
                                
                                .otherwise('Hold'))
    reco_out = reco_out[reco_out['Buy/Not']=='Buy']
    reco_out.toPandas().to_csv('reco_output.csv', index=False)

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
    blob = bucket.blob('data/reco_output.csv')
    blob.upload_from_filename(filename='reco_output.csv')
    print(f"Files are uploaded in the {bucket_name} !!!")

if __name__ == '__main__':
    bucket_name = "bsaw-stock-bucket-1"            # Enter Bucket name here
    ALS_model()
    gcs_bucket(bucket_name)