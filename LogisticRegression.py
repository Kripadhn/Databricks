from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Load and Prepare the Data
data = spark.read.csv("data.csv", header=True, inferSchema=True)
stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
vectorAssembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
pipeline = Pipeline(stages=[stringIndexer, vectorAssembler])
data_indexed = pipeline.fit(data).transform(data)

# Split the Data into Training and Testing Sets
splits = data_indexed.randomSplit([0.7, 0.3])
training_data = splits[0]
testing_data = splits[1]

# Train the Model
lr = LogisticRegression(featuresCol="features", labelCol="indexedLabel", predictionCol="prediction")
model = lr.fit(training_data)

# Evaluate the Model
predictions = model.transform(testing_data)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="indexedLabel")
auc = evaluator.evaluate(predictions)
print("AUC: ", auc)

# Use the Model to Make Predictions
predictions.select("prediction", "indexedLabel", "features").show()
