# Python code for the sepsis project
# Logistic Regression using Apache Spark 
# First import packages and classes that we will need throughout

## Imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col

##OTHER FUNCTIONS/CLASSES
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()

sqlContext = SQLContext(sc)
sepsis = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('sepsis_num.csv')
#sepsis.printSchema()
print sepsis.count()
labelIndexer = StringIndexer(inputCol="infxnqsofa", outputCol="label")
vecAssembler = VectorAssembler(inputCols=["sepsis_antibiotic","antibiotic","immunosupp_class3","RACE_NUM","ETH_NUM","SEXNUM","icd_ind","icd_rank","sepsis_glucocorticoid","treatment_limit","icd9_477_x","icd9_493_x","age_at_enc","icd9_691_x","temp","biologicals","icd9_995_3","bmi","pain_scale","dnr","dnr_treatment_limit","staph","immunosupp_medname","dncpr_dni","icd9_558_3","albuterol","avpu","avpu_old","dnr_dni","immunosupp_class31"],outputCol = "features")

# Split data
(trainingData, testData) = sepsis.randomSplit([0.7, 0.3])

# Decision Tree Classifier
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline = Pipeline(stages=[labelIndexer, vecAssembler, lr])

# Fit the data
model = pipeline.fit(trainingData)

# Predict
predictions = model.transform(testData)
predictions.printSchema()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
predictions.first()
weightedPrecision = evaluator.evaluate(predictions)
print "Model Weighted Precision: ", weightedPrecision

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print "Model Accuracy: ", accuracy

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
fscore = evaluator.evaluate(predictions)
print "Model F1-Score: ", fscore

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
wrecall = evaluator.evaluate(predictions)
print "Model Weighted Recall: ", wrecall 
