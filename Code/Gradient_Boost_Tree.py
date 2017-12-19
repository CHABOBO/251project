from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=[labelIndexer, vecAssembler, gbt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print(accuracy)
#0.989959732858

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

f1 = evaluator.evaluate(predictions)
print(f1)

#0.985172800629

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")

wp = evaluator.evaluate(predictions)
print(wp)

#0.985974308382
