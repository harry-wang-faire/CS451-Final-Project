from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
import numpy
from pyspark.sql import HiveContext
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, log, dense_rank
from pyspark.sql import DataFrameStatFunctions as statFunc
import math
from datetime import datetime, timedelta

import os
import shutil

name_map = {61: 'NWT', 35:'Ontario', 10: 'NL', 11: 'PEI', 
59: 'BC', 47: 'Saskatchewan', 12: 'Nova Scotia', 24: 'Quebec', 
48: 'Alberta', 46: 'Manitoba', 13: 'New Brunswick', 60: 'Yukon', 62: 'Nunavut' }

directory_path = "GBT_regression_test"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
#sqlContext = SQLContext(sc).setConf("spark.sql.shuffle.partitions", "1")
sqlContext = HiveContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

covid_data_directory = 'aggregated_data/result.csv/part-*'

covid_data = sqlContext.read.format('com.databricks.spark.csv')\
  .options(header='true', inferschema='true').load(covid_data_directory)

train_df = covid_data.filter(covid_data.date < 27).filter(covid_data.date > 0)
test_df = covid_data.filter(covid_data.date >= 27)

def gradient_model_generator(train_df, test_df):

  def flatten_features(x):
    f = x.features
    return (x.numconfirmed, int(x.prediction))

  lr = LinearRegression(featuresCol = 'features', labelCol='numconfirmed', maxIter=10, regParam=0.3, elasticNetParam=0.8)
  lr_model = lr.fit(train_df)
  print("Coefficients: " + str(lr_model.coefficients))
  print("Intercept: " + str(lr_model.intercept))

  predictions = lr_model.transform(test_df)

  trainingSummary = lr_model.summary

  print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
  print("r2: %f" % trainingSummary.r2)

  predictions = predictions.rdd.map(flatten_features)
  return predictions

train_df = train_df.withColumn("log_date", log(col('date')))
test_df = test_df.withColumn("log_date", log(col('date')))

train_df.show()
test_df.show()

vectorAssembler = VectorAssembler(inputCols=['log_date','numtested', 'median_age_Scaled', 'population_Scaled'], outputCol = 'features')

train_df = vectorAssembler.transform(train_df)
train_df = train_df.select(['features', 'numconfirmed'])

test_df = vectorAssembler.transform(test_df)
test_df = test_df.select(['features', 'numconfirmed'])

train_df.show()
test_df.show()

result = gradient_model_generator(train_df, test_df)
result_df = spark.createDataFrame(result, ['actual', 'prediction'])
result_df.show()

result_df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(directory_path)


