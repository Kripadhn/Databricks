from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
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
rf = RandomForestClassifier(featuresCol="features", labelCol="indexedLabel", predictionCol="prediction")
model = rf.fit(training_data)

# Evaluate the Model
predictions = model.transform(testing_data)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexedLabel", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: ", accuracy)

# Use the Model to Make Predictions
predictions.select("prediction", "indexedLabel", "features").show()
