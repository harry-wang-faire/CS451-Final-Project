# CS451 Final Project
> This is a comprehensive readme that helps understand and run our project's code

## Installation

The following are required inorder to run our code:
```
pyspark
sparkml

```
## Usage example

for any file end with .py, run
```
spark-submit file.py
```


##  File indexes

* age_extraction.py
    * This file outputs extracted data from canadian population that contains tuple (geo_code, age_group, total population)
* age_training_data_extraction.py
    * This file outputs extracted data from confirmed cases that contains tuple (days since first date, geo_code, age_group, total number of confirmed cases)
* testing_by_province.py
    * This file outputs extracted data from tested cases that contains tuple(days since first date, geo_code, total number of people tested)
