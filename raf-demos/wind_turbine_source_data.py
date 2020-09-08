# Databricks notebook source
import os
import random
from time import sleep
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

data_uri = dbutils.widgets.get("data_uri")

# COMMAND ----------

random_udf = udf(lambda: int(random.random() * 100000), IntegerType()).asNondeterministic()

feature_cols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE"]
csv_schema = StructType([StructField(fcol, DoubleType()) for fcol in feature_cols])

raw_csv_sdf = (
  spark.read
  .csv("/mnt/databricks-datasets-private/ML/wind_turbine", schema=csv_schema)
  .withColumn("key", random_udf())
  .withColumn("ID", monotonically_increasing_id())
  .withColumn("TIMESTAMP", current_timestamp())
  .withColumn("value", to_json(struct(*feature_cols, "ID", "TIMESTAMP")))
  .withColumn("STATUS", substring(reverse(split(input_file_name(), "/"))[0], 1, 1))
  .withColumn("STATUS", when(col("STATUS") == "D", "damaged").otherwise("healthy"))
)

# COMMAND ----------

(
  raw_csv_sdf
  .select("ID", "STATUS")
  .write
  .format("delta")
  .mode("overwrite")
  .save(os.path.join(data_uri, "status"))
)

# COMMAND ----------

(
  raw_csv_sdf
  .select("key", "value")
  .repartition(5000) # small files!
  .write
  .mode("overwrite")
  .save(os.path.join(data_uri, "raw"))
)

# COMMAND ----------

dbutils.notebook.exit("0")

# COMMAND ----------

dbutils.widgets.text("data_uri", "/Users/stuart.lynn@databricks.com/demo/turbine")