from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
sepsisf = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('sepsis_f.csv')
(trainingData, testData) = sepsisf.randomSplit([0.7, 0.3])
labelIndexer = StringIndexer(inputCol="infxnqsofa", outputCol="label").fit(sepsisf)
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=["sepsis_antibiotic","antibiotic","RACE_NUM","ETH_NUM","SEXNUM","icd_ind","icd_rank","sepsis_glucocorticoid","treatment_limit","icd9_477_x","icd9_493_x","age_at_enc","icd9_691_x","temp","biologicals","icd9_995_3","bmi","pain_scale","dnr","dnr_treatment_limit","staph","immunosupp_medname","dncpr_dni","icd9_558_3","albuterol","avpu","avpu_old","dnr_dni","immunosupp_class30"],outputCol = "features")
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
pipeline = Pipeline(stages=[labelIndexer, vecAssembler, rf, labelConverter])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)
#0.989923015099
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print(f1)
#0.984910037612
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluator.evaluate(predictions)
print(wp)
#0.979947575823
