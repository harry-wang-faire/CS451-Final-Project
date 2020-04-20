from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
import numpy
from pyspark.sql import HiveContext
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

import os
import shutil

name_map = {61: 'NWT', 35:'Ontario', 10: 'NL', 11: 'PEI', 
59: 'BC', 47: 'Saskatchewan', 12: 'Nova Scotia', 24: 'Quebec', 
48: 'Alberta', 46: 'Manitoba', 13: 'New Brunswick', 60: 'Yukon', 62: 'Nunavut' }

directory_path = "GBT_regression_result"
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

train_df = covid_data.filter(covid_data.date < 27)
test_df = covid_data.filter(covid_data.date >= 27)

print(train_df.dtypes)
print(test_df.dtypes)


def gradient_model_generator(train_df, test_df):

  def flatten_features(x):
    f = x.features
    return (x.numconfirmed, int(x.prediction), name_map.get(int(f[0])), int(f[1]), int(f[2]), float(f[3]), float(f[4]))

  gbt = GBTRegressor(featuresCol = 'features', labelCol='numconfirmed', maxIter=10)
  gbt_model = gbt.fit(train_df)

  predictions = gbt_model.transform(test_df)

  gbt_evaluator = RegressionEvaluator(
      labelCol="numconfirmed", predictionCol="prediction", metricName="rmse")
  rmse = gbt_evaluator.evaluate(predictions)
  print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

  predictions = predictions.rdd.map(flatten_features)
  return predictions

cSchema = StructType([StructField("numconfirmed", IntegerType())\
                      ,StructField("prediction", IntegerType())\
                        ,StructField("province", StringType())\
                          ,StructField("date", StringType())
                            ,StructField("numtested", IntegerType())\
                              ,StructField("median_age_scaled", FloatType())\
                                ,StructField("population_scaled", FloatType())])

df = spark.createDataFrame([], schema=cSchema)
# iterate over every province's data
for code in name_map.keys():
  province_train_df = train_df.filter(train_df.geo_code == code)
  province_test_df = test_df.filter(test_df.geo_code == code)

  if not len(province_train_df.head(1)) == 0:
    vectorAssembler = VectorAssembler(inputCols=['geo_code','date','numtested', 'median_age_Scaled', 'population_Scaled'], outputCol = 'features')
    
    province_train_df = vectorAssembler.transform(province_train_df)
    province_train_df = province_train_df.select(['features', 'numconfirmed'])

    province_test_df = vectorAssembler.transform(province_test_df)
    province_test_df = province_test_df.select(['features', 'numconfirmed'])

    province_train_df.show()
    province_test_df.show()

    result = gradient_model_generator(province_train_df, province_test_df)
    result_df = spark.createDataFrame(result, schema=cSchema)
    result_df.show()
    df = df.unionAll(result_df)
    df.show()

df.write.csv(directory_path, header=True)


