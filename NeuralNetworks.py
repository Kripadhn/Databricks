from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the data
data = spark.read.format("libsvm").load("/FileStore/tables/sample_libsvm_data.txt")

# Split the data into training and test sets
train, test = data.randomSplit([0.6, 0.4], seed=12345)

# Define the layers of the neural network
layers = [data.shape[1] - 1, 5, 5, 2]

# Train the model
model = MultilayerPerceptronClassifier(layers=layers, maxIter=100, seed=12345)
model = model.fit(train)

# Predict on the test data
predictions = model.transform(test)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)
