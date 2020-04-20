from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from datetime import datetime
import os
import shutil


geocode_map = {'NWT': 61, 'Ontario': 35, 'NL':10, 'PEI':11, 
'BC':59, 'Saskatchewan':47, 'Nova Scotia': 12, 'Quebec' : 24, 
 'Alberta': 48, 'Manitoba': 46, 'New Brunswick': 13, 'Yukon': 60}

profile_set = {
    "0-9": 1,
    "10-19": 1,
    "20-29": 2,
    "30-39": 2,
    "40-49": 3,
    "50-59": 4,
    "60-69": 5,
    "70-79": 6,
    "80-89": 7,
    "90-99": 7,
    "100": 7
}
start_date = datetime(2020, 1, 25)

def map_func(tokens):
    date = datetime.strptime(str(tokens[7]), "%Y-%m-%d")
    delta = date - start_date
    district = geocode_map[tokens[5]]
    age = tokens[2]
    age_id = -1
    if age in profile_set:
        age_id = profile_set[age]
    elif age.find("<") > -1:
        age_id = 1
    else:
        age_id = 7

    return ((delta.days, district, age_id), 1)

directory_path = "age_training_data_extracted"
if (os.path.exists(directory_path)): shutil.rmtree(directory_path)
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder \
    .master("local") \
    .appName("Age Extraction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    

df = spark.read.option("header","true").csv("data/cases.csv")
# text_file = sc.textFile("data/2016Population.csv")
training_data = df.rdd.map(list).filter(lambda tokens: tokens[6] == "Canada" and tokens[5] in geocode_map)\
    .filter(lambda tokens: tokens[2] != "Not Reported")\
    .map(map_func) \
    .reduceByKey(lambda x, y: x+y) \
    .map(lambda tuple: (tuple[0][0], tuple[0][1], tuple[0][2], tuple[1])) \
    .sortBy(lambda tuple: tuple[0])

# age_data = df.rdd.map(list).map(filter_age_func) \
#             .reduceByKey(lambda a, b: a + b) \
#             .sortBy(lambda tuple: tuple[0]) \
#             .map(lambda tuple: (tuple[0][0], tuple[0][1], tuple[1])) 


df = spark.createDataFrame(training_data, ['date', 'geo_code', 'category', 'count'])
df.write.csv(directory_path + "age.csv", header=True)