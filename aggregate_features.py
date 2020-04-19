from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
import numpy
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline

import os
import shutil

name_map = {61: 'NWT', 35:'Ontario', 10: 'NL', 11: 'PEI', 
59: 'BC', 47: 'Saskatchewan', 12: 'Nova Scotia', 24: 'Quebec', 
 48: 'Alberta', 46: 'Manitoba', 13: 'New Brunswick', 60: 'Yukon', 62: 'Nunavut' }

directory_path = "aggregated_data"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sqlContext = SQLContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

covid_data_directory = 'covid19/result.csv/part-*'
features_data_directory = 'data/features.csv'

covid_data = sqlContext.read.format('com.databricks.spark.csv')\
  .options(header='true', inferschema='true').load(covid_data_directory)
features_data = sqlContext.read.format('com.databricks.spark.csv')\
  .options(header='true', inferschema='true').load(features_data_directory)

print("Before Scaling :")
covid_data.show()
features_data.show()

# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

# iterating columns to be normalized
for i in ["median_age", "population"]:
  # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    features_data = pipeline.fit(features_data).transform(features_data).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")


print("After Scaling :")
features_data.show(5)

left_join = covid_data.join(features_data, ['geo_code'], how='left').na.drop()

left_join.show()

left_join.write.csv(directory_path+"/result.csv", header=True)



