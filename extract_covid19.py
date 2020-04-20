from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

import os
import shutil
from datetime import datetime

geocode_map = {'Northwest Territories': 61, 'Ontario': 35, 'Newfoundland and Labrador':10, 'Prince Edward Island':11, 
'British Columbia':59, 'Saskatchewan':47, 'Nova Scotia': 12, 'Quebec' : 24, 
 'Alberta': 48, 'Manitoba': 46, 'New Brunswick': 13, 'Yukon': 60, 'Nunavut':62 }

start_date = datetime(2020,3,15)

directory_path = "covid19"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

def map_func(tokens):
    date = datetime.strptime(str(tokens[3]), "%d-%m-%Y")
    geocode = geocode_map.get(tokens[1]) or None
    confirmed_cases = int(tokens[7]) if (tokens[7] is not None) else 0
    num_tested = int(tokens[8]) if (tokens[8] is not None) else 0
    delta = date - start_date
    return (geocode, delta.days, confirmed_cases, num_tested)
    

df = spark.read.option("header","false").csv("data/covid.csv")
training_data = df.rdd.map(list).map(map_func)\
    .filter(lambda x: x[0] is not None and x[1] >= 0)\
    .sortByKey()


# convert rdd to dataframe
df = spark.createDataFrame(training_data, ['geo_code', 'date', 'numconfirmed', 'numtested'])
df.write.csv(directory_path + "/result.csv", header=True)