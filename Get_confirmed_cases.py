from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.linalg import DenseVector
import numpy as np

import os
import shutil
from datetime import datetime

start_date = datetime(2020,3,15)
input_data_start_date = datetime(2020,1,22)
col_to_remove = int((start_date - input_data_start_date).days) + 4 - 1 # 4 non-case column

input_path = "data/time_series_covid19_confirmed_global.csv"
output_path = "cases"

geocode_map = {'Northwest Territories': 61, 'Ontario': 35, 'Newfoundland and Labrador':10, 'Prince Edward Island':11,
'British Columbia':59, 'Saskatchewan':47, 'Nova Scotia': 12, 'Quebec' : 24,
 'Alberta': 48, 'Manitoba': 46, 'New Brunswick': 13, 'Yukon': 60, 'Nunavut':62 }

if (os.path.exists(output_path)): shutil.rmtree(output_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sqlContext = SQLContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Training Prediction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


df = spark.read.option("header","true").csv(input_path) \
                .withColumnRenamed('Province/State', 'province')

#(province, list of confirm cases by day starting from start_date)
data = df.filter(df.province.isin(geocode_map.keys())) \
        .rdd.map(list) \
        .map(lambda x : ((geocode_map.get(x[0]) , x[0]), [int(y) for y in x[col_to_remove:]])) \
        .collect()

for d in data:
    cases = sc.parallelize(list(zip(d[1][1:], np.diff(d[1])))) \
    .zipWithIndex() \
    .map(lambda x: (x[0][0], x[0][1], x[1])) \
    .repartition(1)
    # (number of cases, number of new cases, number of days since start day)
    cases.saveAsTextFile(output_path + "/" + d[0][1])
