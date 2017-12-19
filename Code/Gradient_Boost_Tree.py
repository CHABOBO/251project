from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#RUN with 2 mil non-septic and 24000
#RUN with 1 mil non-septic and 24000
###WILL NEED TO RERUN ONCE FEATURES HAVE BEEN TUNED TO PATIENT LEVEL###

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=[labelIndexer, vecAssembler, gbt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print(accuracy)
#0.989959732858
#0.976641172621
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

f1 = evaluator.evaluate(predictions)
print(f1)

#0.985172800629
#0.965841974335
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")

wp = evaluator.evaluate(predictions)
print(wp)

#0.985974308382
#0.969283108931

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

rc = evaluator.evaluate(predictions)
print(rc)
#0.98999361005
#0.976641172621
