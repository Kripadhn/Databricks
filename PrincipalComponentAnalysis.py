from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

# Define the data
data = [(Vectors.dense([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([0.1, 1.1, 2.1, 3.1, 4.1, 5.1]),),
        (Vectors.dense([0.2, 1.2, 2.2, 3.2, 4.2, 5.2]),),
        (Vectors.dense([0.3, 1.3, 2.3, 3.3, 4.3, 5.3]),)]
df = spark.createDataFrame(data, ["features"])

# Train the PCA model
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
model = pca.fit(df)

# Transform the data
result = model.transform(df)
result.show()
