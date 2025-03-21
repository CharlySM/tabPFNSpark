from pyspark.sql import SparkSession
import pandas as pd
from sklearn.metrics import accuracy_score
from pyspark.sql.functions import spark_partition_id
# Import Pipeline and MinMaxScaler
from pyspark.ml.feature import MinMaxScaler, StringIndexer

from Utils import train_and_evaluate

# Inicializar Spark
spark = SparkSession.builder.appName("TabPFNTraining").getOrCreate()
sc = spark.sparkContext
spark.sparkContext.setLogLevel('WARN')

# Cargar datos a un pandas dataframe.

# Dividir datos en entrenamiento y prueba.
dfInit, test=spark.read.option("inferSchema","true").option("header","true") \
    .csv("/content/Student Depression Dataset.csv").dropna().withColumnRenamed("Depression", "label") \
    .randomSplit([0.75,0.25])

columns=result = result = [a for a in dfInit.columns if a not in "label"]
colsIndex = [x + "Index" for x in columns]
mappCols = {x + "Index": x for x in columns}

stringIndexer = StringIndexer(inputCols=columns, outputCols=colsIndex)
dfScalerTrain = stringIndexer.fit(dfInit).transform(dfInit).drop(*columns) \
    .withColumnsRenamed(mappCols)
test = stringIndexer.fit(test).transform(test).drop(*columns) \
    .withColumnsRenamed(mappCols)

dfScalerTrain.printSchema()
countTrain=dfScalerTrain.count()
coundTest=test.count()
print(countTrain,coundTest)

global count
count=0
predictions=[]

# Convert test RDD to a pandas DataFrame outside the mapPartitions call
test_df = test.repartition(10).withColumn("partition_id", spark_partition_id() % 10).toPandas() # Convert test RDD to pandas DataFrame
# Entrenar modelos en paralelo.
dfScalerTrain=dfScalerTrain.repartition(10).withColumn("partition_id", spark_partition_id() % 10)
columns=dfScalerTrain.columns
trained_models = (dfScalerTrain.repartition(10).rdd \
    .mapPartitions(lambda partition: train_and_evaluate(partition, columns, test_df)) \
    .filter(lambda x: x is not None))

for i in trained_models.collect():
    predictions.extend(i)
print(predictions)

trained_models.show()

y_test = test_df["label"]
min_len = min(len(y_test), len(predictions))  # Get the minimum length
y_test = y_test[:min_len]                    # Truncate y_test
predictions = predictions[:min_len]

accuracy = accuracy_score(y_test, predictions)

print("accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix

tp, fn, fp, tn = confusion_matrix(y_test, predictions).ravel()
TPR = tp/(tp+fn)
TNR = tn/(tn+fp)
score = TPR * TNR

print(TPR, TNR, score)
