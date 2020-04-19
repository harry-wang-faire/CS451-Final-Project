from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import os
import shutil



profile_set = {
    "0 to 4 years": 1,
    "5 to 9 years": 1,
    "10 to 14 years": 1,
    "15 to 19 years": 1,
    "20 to 24 years": 2,
    "25 to 29 years": 2,
    "30 to 34 years": 2,
    "35 to 39 years": 2,
    "40 to 44 years": 3,
    "45 to 49 years": 3,
    "50 to 54 years": 4,
    "55 to 59 years": 4,
    "60 to 64 years": 5,
    "65 to 69 years": 5,
    "70 to 74 years": 6,
    "75 to 79 years": 6,
    "80 to 84 years": 7,
    "85 to 89 years": 7,
    "90 to 94 years": 7,
    "95 to 99 years": 7,
    "100 years and over": 7
}
def filter_age_func(tokens):
    # age groups: children: 0-9
    #             youth:
    geo_code = int(tokens[1])
    population_type = profile_set[tokens[8]]
    population_total = int(tokens[11])
    return ((geo_code, population_type), population_total)

directory_path = "age_data_extracted"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    

df = spark.read.option("header","true").csv("data/2016Population.csv")
# text_file = sc.textFile("data/2016Population.csv")
age_data = df.rdd.map(list).filter(lambda list: list[8] in profile_set) \
            .map(filter_age_func) \
            .reduceByKey(lambda a, b: a + b) \
            .sortBy(lambda tuple: tuple[0]) \
            .map(lambda tuple: (tuple[0][0], tuple[0][1], tuple[1])) 


df = spark.createDataFrame(age_data)
df.write.format("csv").save(directory_path)