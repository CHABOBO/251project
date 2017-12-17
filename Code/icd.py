# Python code for the sepsis project
# Logistic Regression using Apache Spark 
# First import packages and classes that we will need throughout

## Imports

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from operator import add
import sys

## Constants
APP_NAME = "ICD APP"

##OTHER FUNCTIONS/CLASSES

def main(sc,spark,filename,col_index):
   icddf=spark.read.format("csv").option("header", "true").load(filename)

   # Check to makes sure the data set is loaded correctly
   #icddf.show()
   #icddf.printSchema()
   #print(icddf.count()) 
   assert icddf.count()==15119485

   # Read few lines

   print icddf.take(1)
   icddf.select("infxnqsofa").show()
   icddf.select("temp").show()
   #uval = icddf.rdd.map(lambda x: x[0]).distinct().collect()
   #print uval
   # given a list of indicies...
   indicies = [int(i) for i in [1,2]]

   # select only those columns from each row
   idf = icddf.rdd.map(lambda x: [x[idx] for idx in indicies])

   idf.show()


   
if __name__ == "__main__":

   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME)
   conf = conf.setMaster("local[*]")
   sc   = SparkContext(conf=conf)
   spark = SparkSession.builder.master("local").appName(APP_NAME).getOrCreate()
   filename = sys.argv[1]
   input_col_index = sys.argv[2] # For example - ['1','2','3','4']
   # Execute Main functionality
   main(sc, spark, filename, input_col_index)
