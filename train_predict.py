from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.linalg import DenseVector

import os
import shutil
from datetime import datetime

# run Get_confirmed_cases.py first

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

def trainbyprovince(trainingData, testData):
    glr = GeneralizedLinearRegression(family="gaussian", maxIter=10, regParam=0.3)
    model = glr.fit(trainingData)

    predictions = model.transform(testData)
    output = predictions.select("prediction", "label", "features").take(2)
    return output

for dir in os.walk(input_folder):
    if dir[0] == input_folder:
        continue
    input_path = dir[0] + file_name
    output_path = dir[0] + output_file


    df = sc.textFile(input_path)
    # we want to run the model on each province
    data = df.map(lambda x: [int(y) for y in x.strip('()').split(",")])

    trainingData = sc.parallelize(data.take(data.count() - 2)) \
        .map(lambda x : Row(label = x[0], features = DenseVector([x[1]]))) \
        .toDF()

    testData = sc.parallelize(data.map(lambda (a, b): (b, a)).top(2)) \
        .map(lambda x : Row(label = x[1], features = DenseVector([x[0]]))) \
        .toDF()

    result = trainbyprovince(trainingData, testData)

    f = open(output_path, 'w')
    f.write(str(result))
    f.close()
