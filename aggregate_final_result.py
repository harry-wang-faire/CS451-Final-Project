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


directory_path = "final_prediction"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
#sqlContext = SQLContext(sc).setConf("spark.sql.shuffle.partitions", "1")
sqlContext = HiveContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

covid_data_directory = 'GBT_regression_result/part-*'

df = sqlContext.read.format('com.databricks.spark.csv')\
  .options(header='true', inferschema='true').load(covid_data_directory)

print(df.schema)
df = df.groupBy("date").agg({'prediction':'sum'}).orderBy('date')
df.show()
df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(directory_path)


