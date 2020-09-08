# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/turbine/turbine_flow.png" />
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

import random
from time import sleep
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

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
  .save("/Users/stuart.lynn@databricks.com/demo/turbine/status")
)

# COMMAND ----------

(
  raw_csv_sdf
  .select("key", "value")
  .repartition(5000) # small files!
  .write
  .mode("overwrite")
  .csv("/Users/stuart.lynn@databricks.com/demo/turbine/raw/")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Bronze layer: ingest data from streaming source (could be Kafka, Kinesis, Event Hubs)

# COMMAND ----------

bronzeDF = (
    spark.readStream
#         .format("kafka")
#         .option("kafka.bootstrap.servers", "kafkaserver1:9092,kafkaserver2:9092")
#         .option("subscribe", "turbine")
        .option("maxFilesPerTrigger", 1)
        .schema("key string, value string")
        .csv("/Users/stuart.lynn@databricks.com/demo/turbine/raw/")
)

bronzeDF.display()

# COMMAND ----------

# Write the output to a delta table
(
    bronzeDF
        .writeStream
        .format("delta")
        .option("checkpointLocation", "/Users/stuart.lynn@databricks.com/demo/turbine/bronze/_checkpoint")
        .option("path", "/Users/stuart.lynn@databricks.com/demo/turbine/bronze/data")
        .start()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add the table in our data catalog
# MAGIC create table if not exists stuart.turbine_bronze
# MAGIC   using delta
# MAGIC   location '/Users/stuart.lynn@databricks.com/demo/turbine/bronze/data';
# MAGIC 
# MAGIC -- Turn on autocompaction to solve small files issues on your streaming job, that's all you have to do!
# MAGIC alter table stuart.turbine_bronze set tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);
# MAGIC 
# MAGIC -- Select data
# MAGIC select * from stuart.turbine_bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select data
# MAGIC select count(*) from stuart.turbine_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Silver layer: transform JSON data into tabular table

# COMMAND ----------

jsonSchema = StructType([StructField(col, DoubleType(), False) for col in
                         ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]] + [
                            StructField("TIMESTAMP", TimestampType())])

silverStream = spark.readStream.table('stuart.turbine_bronze')
(
    silverStream
        .withColumn("jsonData", from_json(col("value"), jsonSchema))
        .select("jsonData.*")
        .writeStream
        .format("delta")
        .trigger(once=True)
        .option("checkpointLocation", "/Users/stuart.lynn@databricks.com/demo/turbine/silver/_checkpoint")
        .option("path", "/Users/stuart.lynn@databricks.com/demo/turbine/silver/data")
        .start()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add the table in our data catalog
# MAGIC create table if not exists stuart.turbine_silver
# MAGIC   using delta
# MAGIC   location '/Users/stuart.lynn@databricks.com/demo/turbine/silver/data';
# MAGIC 
# MAGIC -- Select data
# MAGIC select * from stuart.turbine_silver;

# COMMAND ----------

(
    spark.read
        .format("delta")
        .load("/Users/stuart.lynn@databricks.com/demo/turbine/status")
        .display()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Gold layer: join information on damaged table to add a label to our dataset

# COMMAND ----------

turbine_stream = spark.readStream.table('stuart.turbine_silver')
turbine_status = spark.read.format("delta").load("/Users/stuart.lynn@databricks.com/demo/turbine/status")

(
    turbine_stream
        .join(turbine_status, ['ID'], 'left')
        .writeStream
        .format("delta")
        .trigger(once=True)
        .option("checkpointLocation", "/Users/stuart.lynn@databricks.com/demo/turbine/gold/_checkpoint")
        .option("path", "/Users/stuart.lynn@databricks.com/demo/turbine/gold/data")
        .start()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add the table in our data catalog
# MAGIC create table if not exists stuart.turbine_gold
# MAGIC   using delta
# MAGIC   location '/Users/stuart.lynn@databricks.com/demo/turbine/gold/data';
# MAGIC 
# MAGIC -- Select data
# MAGIC select * from stuart.turbine_gold;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run DELETE/UPDATE/MERGE with DELTA ! 
# MAGIC We just realized that something is wrong in our data - we cannot measure TORQUE > 170! Let's DELETE all this data from our gold table as we don't want to have wrong value in our dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM stuart.turbine_gold where TORQUE > 170

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Use ML and MLFlow to detect damaged turbine
# MAGIC 
# MAGIC Our data is now ready. We'll now train a model to detect damaged turbines.

# COMMAND ----------

from pyspark.sql.functions import rand

dataset = spark.read.table("stuart.turbine_gold")
dataset = dataset.orderBy(rand()).cache()
turbine_healthy = dataset.filter("STATUS = 'healthy'")
turbine_damaged = dataset.filter("STATUS = 'damaged'")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 
# MAGIC 
# MAGIC *Notice the much larger stdev in AN8, AN9 and AN10 for Damaged turbines.*

# COMMAND ----------

# Healthy turbine
display(turbine_healthy.describe())

# COMMAND ----------

# Damaged turbine
display(turbine_damaged.describe())

# COMMAND ----------

# Compare AN9 value for healthy/damaged; varies much more for damaged ones
display(dataset)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: Workflows with Pyspark.ML Pipeline

# COMMAND ----------

# DBTITLE 1,Build Training and Test dataset
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

train, test = dataset.limit(1000000).randomSplit([0.8, 0.2])
print(train.count())
print(test.count())

# COMMAND ----------

# DBTITLE 1,Train our model using a GBT
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow.spark
import mlflow

with mlflow.start_run():

    gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
    grid = ParamGridBuilder().addGrid(gbt.maxDepth, [4, 5, 6]).build()

    ev = BinaryClassificationEvaluator()

    # 3-fold cross validation
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=ev, numFolds=3)

    featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
    stages = [VectorAssembler(inputCols=featureCols, outputCol="va"),
              StandardScaler(inputCol="va", outputCol="features"), 
              StringIndexer(inputCol="STATUS", outputCol="label"),
              cv]
    pipeline = Pipeline(stages=stages)

    pipelineTrained = pipeline.fit(train)

    mlflow.spark.log_model(pipelineTrained, "turbine_gbt")
    mlflow.set_tag("model", "turbine_gbt")
    predictions = pipelineTrained.transform(test)
    # Prints AUROC
    AUROC = ev.evaluate(predictions)
    mlflow.log_metric("AUROC", AUROC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Model

# COMMAND ----------

predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql SELECT avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) AS accuracy FROM predictions

# COMMAND ----------

bestModel = pipelineTrained.stages[-1:][0].bestModel
# convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF(
    "weight", "feature")
weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False).display()

# COMMAND ----------

# MAGIC %md ## Saving our model to MLFLow registry

# COMMAND ----------

# DBTITLE 1,Save our new model to the registry as a new version

# get the best model having the best metrics.AUROC from the registry
best_models = mlflow.search_runs(
    filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.AUROC > 0').sort_values(
    by=['metrics.AUROC'], ascending=False)
model_uri = best_models.iloc[0].artifact_uri

model = mlflow.register_model(best_models.iloc[0].artifact_uri + "/turbine_gbt", "turbine_gbt_sl")
print("Model  " + str(model.version) + " has been registered!")
sleep(5)

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
client = mlflow.tracking.MlflowClient()
########
# NOTE: this won't be necessary in the next client release (use archive_existing_versions instead)
# for old_model_in_production in client.get_latest_versions(name="turbine_gbt_sl", stages=["Production"]):
#     if old_model_in_production.version != model.version:
#         client.update_model_version(name="turbine_gbt_sl", version=old_model_in_production.version, stage="Archived")
#########

client.transition_model_version_stage(name="turbine_gbt_sl", version=model.version, stage="Production")  # NOTE: add here archive_existing_versions=true with new client version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detecting damaged turbine in a production pipeline

# COMMAND ----------

# DBTITLE 1,Load the model from our registry
model_from_registry = mlflow.spark.load_model('models:/turbine_gbt_sl/production')

# COMMAND ----------

# DBTITLE 1,Compute predictions using our spark model:
prediction = model_from_registry.transform(dataset)
prediction.select(*featureCols + ['prediction']).display()