from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

# Define the data
data = [(Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([1.0, 1.0]),),
        (Vectors.dense([9.0, 8.0]),),
        (Vectors.dense([8.0, 9.0]),),]
df = spark.createDataFrame(data, ["features"])

# Train the KMeans model
kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(df)

# Predict the cluster for each data point
predictions = model.transform(df)
predictions.show()
