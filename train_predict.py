from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.linalg import DenseVector

import os
import shutil
from datetime import datetime

# run Get_confirmed_cases.py first
# train on daily new case data

predict_day = datetime(2020,4,30)
start_date = datetime(2020,3,15)

input_folder = "cases"
file_name = "/part-00000"
output_file = "/results.txt"

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sqlContext = SQLContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Training Prediction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

def train(trainingData, testData):
    glr = GeneralizedLinearRegression(family="gaussian", maxIter=10, regParam=0.3)
    model = glr.fit(trainingData)

    predictions = model.transform(testData)
    output = predictions.select("prediction", "label", "features").take(2)
    newcases = predictions.select("prediction").groupBy().sum().collect()[0].__getitem__("sum(prediction)")
    return output, newcases

canada_total_cases_three_days_ago = 0
canada_total_cases_most_recent = 0
canada_total_cases_most_recent_predict = 0

for dir in os.walk(input_folder):
    if dir[0] == input_folder:
        continue
    input_path = dir[0] + file_name
    output_path = dir[0] + output_file


    df = sc.textFile(input_path)
    # we want to run the model on each province
    data = df.map(lambda x: [int(y) for y in x.strip('()').split(",")]) \
            .map(lambda x : (x[1], x[2]))

    trainingData = sc.parallelize(data.take(data.count() - 3)) \
        .map(lambda x : Row(label = x[0], features = DenseVector([x[1]]))) \
        .toDF()

    testData = sc.parallelize(data.map(lambda (a, b): (b, a)).top(3)[-2:]) \
        .map(lambda x : Row(label = x[1], features = DenseVector([x[0]]))) \
        .toDF()

    (result, newcases) = train(trainingData, testData)

    f = open(output_path, 'w')
    f.write(str(result))
    f.close()

    p_total_case = df.map(lambda x: [int(y) for y in x.strip('()').split(",")]) \
                                            .map(lambda x: x[0]).collect()[-4:]
    canada_total_cases_three_days_ago += p_total_case[0]
    canada_total_cases_most_recent += p_total_case[-1]
    canada_total_cases_most_recent_predict += p_total_case[0] + newcases

f = open(input_folder + output_file, 'w')
f.write("canada_total_cases_three_days_ago: " +  str(canada_total_cases_three_days_ago) + "\n")
f.write("canada_total_cases_most_recent: " +  str(canada_total_cases_most_recent) + "\n")
f.write("canada_total_cases_most_recent_predict: " +  str(canada_total_cases_most_recent_predict) + "\n")
f.close()
