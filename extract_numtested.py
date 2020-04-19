from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext


import os
import shutil
from datetime import datetime

geocode_map = {'NWT': 61, 'Ontario': 35, 'NL':10, 'PEI':11, 
'BC':59, 'Saskatchewan':47, 'Nova Scotia': 12, 'Quebec' : 24, 
 'Alberta': 48, 'Manitoba': 46, 'New Brunswick': 13, 'Yukon': 60, 'Nunavut':62 }

start_date = datetime(2020,3,15)

def map_func(tokens):
    date = datetime.strptime(str(tokens[0]), "%Y-%m-%d")
    geocode = geocode_map[tokens[1]]
    confirmed_cases = int(tokens[2]) if (tokens[2] is not None) else 0
    delta = date - start_date
    return (delta.days, geocode, confirmed_cases)

directory_path = "confirmed_cases_data"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    

df = spark.read.option("header","false").csv("data/testing.csv")
# text_file = sc.textFile("data/2016Population.csv")
training_data = df.rdd.map(list).map(map_func)

# convert rdd to dataframe
df = spark.createDataFrame(training_data)
df.write.format("csv").save(directory_path)