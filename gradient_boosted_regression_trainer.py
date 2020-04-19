from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
import numpy
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

import os
import shutil

name_map = {61: 'NWT', 35:'Ontario', 10: 'NL', 11: 'PEI', 
59: 'BC', 47: 'Saskatchewan', 12: 'Nova Scotia', 24: 'Quebec', 
 48: 'Alberta', 46: 'Manitoba', 13: 'New Brunswick', 60: 'Yukon', 62: 'Nunavut' }

directory_path = "GBT_regression_result"
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

# train_df = train_df.select(['features', 'numconfirmed'])
vectorAssembler = VectorAssembler(inputCols=['geo_code','date', 'numtested'], outputCol = 'features')

train_df = vectorAssembler.transform(train_df)
train_df.show()
train_df = train_df.select(['features', 'numconfirmed'])

test_df = vectorAssembler.transform(test_df)
test_df.show()
test_df = test_df.select(['features', 'numconfirmed'])


def flatten_features(x):
  f = x.features
  return (x.numconfirmed, int(x.prediction), name_map.get(int(f[0])), int(f[1]), int(f[2]))

gbt = GBTRegressor(featuresCol = 'features', labelCol='numconfirmed', maxIter=10)
gbt_model = gbt.fit(train_df)

predictions = gbt_model.transform(test_df)

gbt_evaluator = RegressionEvaluator(
    labelCol="numconfirmed", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

predictions = predictions.rdd.map(flatten_features)

df = spark.createDataFrame(predictions, ['numconfirmed', 'prediction', 'province', 'date', 'numtested'])
df.write.csv(directory_path, header=True)

