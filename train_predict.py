from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.linalg import DenseVector

import os
import shutil
from datetime import datetime


predict_day = datetime(2020,4,30)
start_date = datetime(2020,3,15)

input_path = "confirmed_cases_data/part-00000"
output_path = "results.txt"
# if (os.path.exists(output_path)): shutil.rmtree(output_path)

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
sqlContext = SQLContext(sc)
spark = SparkSession.builder \
    .master("local") \
    .appName("Training Prediction") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


def trainbyprovince(data):
    # to DF for training
    mdata = sc.parallelize(data[1]) \
    .map(lambda x : Row(label = x[2], features = DenseVector([x[0]])))

    dataDf = sqlContext.createDataFrame(mdata)
    (trainingData, testData) = dataDf.randomSplit([0.7, 0.3])

    glr = GeneralizedLinearRegression(family="gaussian", maxIter=10, regParam=0.3)
    model = glr.fit(trainingData)

    predictions = model.transform(testData)
    output = predictions.select("prediction", "label", "features").take(2)
    return (data[0], output)



df = sc.textFile(input_path)
# we want to run the model on each province
provinces = df.map(lambda x: [int(y) for y in x.strip('()').split(",")]) \
                .groupBy(lambda x : x[1]) \
                .collect()

f = open(output_path, 'w')
for p in provinces:
    result = trainbyprovince(p)
    f.write(str(result) + "\n")

f.close()

# d = trainbyprovince(predicts.take(1)[0])
# print(d)
# predicts.saveAsTextFile(output_path)
