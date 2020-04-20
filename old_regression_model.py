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

train_df = covid_data.filter(covid_data.numconfirmed > 0)

# change train_df type
train_df = train_df.withColumn("population", train_df["population"].cast(FloatType()))
train_df = train_df.withColumn("population_Scaled", train_df["population_Scaled"].cast(FloatType()))
train_df = train_df.withColumn("median_age", train_df["median_age"].cast(FloatType()))
train_df = train_df.withColumn("median_age_Scaled", train_df["median_age_Scaled"].cast(FloatType()))
train_df.show()


def gradient_model_generator(code, train_df, test_df):

  def flatten_features(x):
    f = x.features
    #print(code, f[0], f[1], f[2])
    return (int(math.exp(x.log_numconfirmed)), int(math.exp(x.prediction)), name_map.get(int(code)), int(f[0]), int(f[1]), float(f[2]), float(f[3]))

  #gbt = GBTRegressor(featuresCol = 'features', labelCol='numconfirmed', maxIter=10)
  #gbt_model = gbt.fit(train_df)

  lr = LinearRegression(featuresCol = 'features', labelCol='log_numconfirmed', maxIter=10, regParam=0.3, elasticNetParam=0.8)
  lr_model = lr.fit(province_train_df)
  print("Coefficients: " + str(lr_model.coefficients))
  print("Intercept: " + str(lr_model.intercept))

  trainingSummary = lr_model.summary
  print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
  print("r2: %f" % trainingSummary.r2)

  predictions = lr_model.transform(test_df)

  #predictions = gbt_model.transform(test_df)

  # gbt_evaluator = RegressionEvaluator(
  #     labelCol="numconfirmed", predictionCol="prediction", metricName="rmse")
  # rmse = gbt_evaluator.evaluate(predictions)
  # print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
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
  province_train_df = province_train_df.withColumn("log_numconfirmed", log(col('numconfirmed').cast(FloatType())))
  #province_test_df = test_df.filter(test_df.geo_code == code)

  if not len(province_train_df.head(1)) == 0:
    # population_train_df = province_train_df.withColumn("log_date", province_train_df["log_date"].cast(FloatType()))

    my_window = Window.partitionBy().orderBy("date")
    province_train_df = province_train_df.withColumn("prev_num_tested", F.lag(province_train_df.numtested).over(my_window))
    province_train_df = province_train_df.withColumn("diff", F.when(F.isnull(province_train_df.numtested - province_train_df.prev_num_tested), 0)
                              .otherwise(province_train_df.numtested - province_train_df.prev_num_tested))
    province_train_df.show()

    # calculate median
    median_num_tested = statFunc(province_train_df).approxQuantile( "diff", [0.5], 0.25)[0]
    max_num_tested = province_train_df.agg({"numtested": "max"}).collect()[0][0]
    max_date = province_train_df.agg({"date": "max"}).collect()[0][0]
    median_age = province_train_df.first().median_age
    population = province_train_df.first().population
    median_age_scaled = province_train_df.first().median_age_Scaled
    population_scaled = province_train_df.first().population_Scaled

    print(median_num_tested, max_num_tested, province_train_df.schema)
    province_test_df = spark.createDataFrame([], schema=province_train_df.schema)
    # build testing data
    prev_num_tested = max_num_tested
    for i in range(1, 42):
      numtested = int(max_num_tested + i * median_num_tested)
      data = [(code, max_date + i, 0, numtested, median_age, population, median_age_scaled, population_scaled, float(0), prev_num_tested, int(median_num_tested))]
      data_rdd = sc.parallelize(data)
      prev_num_tested = numtested
      print(province_train_df.schema)
      temp = spark.createDataFrame(data_rdd, schema=province_train_df.schema)
      province_test_df = province_test_df.unionAll(temp)
    
    province_test_df.show()


    vectorAssembler = VectorAssembler(inputCols=['date','numtested', 'median_age_Scaled', 'population_Scaled'], outputCol = 'features')
    
    province_train_df = vectorAssembler.transform(province_train_df)
    province_train_df = province_train_df.select(['features', 'log_numconfirmed'])

    province_test_df = vectorAssembler.transform(province_test_df)
    province_test_df = province_test_df.select(['features', 'log_numconfirmed'])

    province_train_df.show()
    province_test_df.show()

    result = gradient_model_generator(code, province_train_df, province_test_df)
    result_df = spark.createDataFrame(result, schema=cSchema)
    result_df.show()
    df = df.unionAll(result_df)

start_date = datetime(2020,4,18)

def convert_date_time(x):
  day = int(x.rank)
  date_time = start_date + timedelta(days=day)
  return (x.province, x.prediction, date_time, x.numtested, x.median_age_scaled, x.population_scaled)

# change df
df = df.withColumn(
  "rank", dense_rank().over(Window.partitionBy("province").orderBy("prediction")))
df.show()
final_result = df.rdd.map(convert_date_time)

final = spark.createDataFrame(final_result, ['province', 'prediction', 'date', 'numtested', 'median_age_scaled', 'population_scaled'])
final.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(directory_path)


