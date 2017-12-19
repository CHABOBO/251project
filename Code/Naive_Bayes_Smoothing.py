#cross validate
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).build()
cvEvaluatoracc = MulticlassClassificationEvaluator(metricName="accuracy")
cvacc = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=cvEvaluatoracc)
cvModelacc = cvacc.fit(trainingData)
cvPredictionsacc = cvModelacc.transform(testData)
evaluator.evaluate(cvPredictionsacc)
#0.8989933951745603 
cvEvaluatorf = MulticlassClassificationEvaluator(metricName="f1")
cvf = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=cvEvaluatorf)
cvModelf = cvf.fit(trainingData)
cvPredictionsf = cvModelf.transform(testData)
evaluator.evaluate(cvPredictionsf)
#0.8989933951745603
cvEvaluatorwp = MulticlassClassificationEvaluator(metricName="weightedPrecision")
cvwp = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=cvEvaluatorwp)
cvModelwp = cvwp.fit(trainingData)
cvPredictionswp = cvModelwp.transform(testData)
evaluator.evaluate(cvPredictionswp)
#0.8989933951745603
