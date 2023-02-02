from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Load and Prepare the Data
data = spark.read.csv("data.csv", header=True, inferSchema=True)
vectorAssembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
data_vector = vectorAssembler.transform(data)

# Split the Data into Training and Testing Sets
splits = data_vector.randomSplit([0.7, 0.3])
training_data = splits[0]
testing_data = splits[1]

# Train the Model
lr = LinearRegression(featuresCol='features', labelCol='label')
model = lr.fit(training_data)

# Evaluate the Model
evaluation_results = model.evaluate(testing_data)
print("RMSE: ", evaluation_results.rootMeanSquaredError)
print("R2: ", evaluation_results.r2)

# Use the Model to Make Predictions
predictions = model.transform(testing_data)
predictions.select("prediction", "label", "features").show()
