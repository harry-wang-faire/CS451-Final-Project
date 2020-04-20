from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
import numpy
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

import os
import shutil

name_map = {61: 'NWT', 35:'Ontario', 10: 'NL', 11: 'PEI', 
59: 'BC', 47: 'Saskatchewan', 12: 'Nova Scotia', 24: 'Quebec', 
48: 'Alberta', 46: 'Manitoba', 13: 'New Brunswick', 60: 'Yukon', 62: 'Nunavut' }


directory_path = "linear_regression_result"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sqlContext = SQLContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

covid_data_directory = 'aggregated_data/result.csv/part-*'

covid_data = sqlContext.read.format('com.databricks.spark.csv')\
  .options(header='true', inferschema='true').load(covid_data_directory)

train_df = covid_data.filter(covid_data.date < 27)
test_df = covid_data.filter(covid_data.date >= 27)

# vectorAssembler = VectorAssembler(inputCols=['geo_code','date'], outputCol = 'features')
vectorAssembler = VectorAssembler(inputCols=['geo_code','date','numtested', 'median_age_Scaled', 'population_Scaled'], outputCol = 'features')

train_df = vectorAssembler.transform(train_df)
train_df = train_df.select(['features', 'numconfirmed'])
train_df.show()

test_df = vectorAssembler.transform(test_df)
test_df = test_df.select(['features', 'numconfirmed'])

def flatten_features(x):
  f = x.features
  return (x.numconfirmed, int(x.prediction), name_map.get(int(f[0])), int(f[1]), int(f[2]), float(f[3]), float(f[4]))

lr = LinearRegression(featuresCol = 'features', labelCol='numconfirmed', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

predictions = lr_model.transform(test_df)
predictions = predictions.rdd.map(flatten_features)

# convert rdd to dataframe
df = spark.createDataFrame(predictions, ['numconfirmed', 'prediction', 'province', 'date', 'numtested', 'median_age_scaled', 'population_scaled'])
df.write.csv(directory_path, header=True)

