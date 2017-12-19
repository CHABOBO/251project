
#pyspark
from pyspark import SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sqlContext = SQLContext(sc)
sepsis = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('sepsis_num.csv')
sepsis.printSchema()
labelIndexer = StringIndexer(inputCol="infxnqsofa", outputCol="label")
(trainingData, testData) = sepsis.randomSplit([0.7, 0.3])
trainingData.cache()
testData.cache()
print trainingData.count()
print testData.count()
print sepsis.count()
vecAssembler = VectorAssembler(inputCols=["sepsis_antibiotic","antibiotic","immunosupp_class3","RACE_NUM","ETH_NUM","SEXNUM","icd_ind","icd_rank","sepsis_glucocorticoid","treatment_limit","icd9_477_x","icd9_493_x","age_at_enc","icd9_691_x","temp","biologicals","icd9_995_3","bmi","pain_scale","dnr","dnr_treatment_limit","staph","immunosupp_medname","dncpr_dni","icd9_558_3","albuterol","avpu","avpu_old","dnr_dni","immunosupp_class31"],outputCol = "features")
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
pipeline = Pipeline(stages=[labelIndexer, vecAssembler, nb])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.printSchema()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightPrecision")
predictions.first()
accuracy = evaluator.evaluate(predictions)
print "Model Weighted Precision: ", weightedPrecision
#Model Weighted Precision:  0.988199872338
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print "Model Accuracy: ", accuracy
#Model Accuracy:  0.831045640258
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
fscore = evaluator.evaluate(predictions)
print "Model F1-Score: ", fscore
#Model F1-Score: 0.898684148073
