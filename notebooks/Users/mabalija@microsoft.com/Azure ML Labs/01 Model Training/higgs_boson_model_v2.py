# Databricks notebook source
from pyspark.sql.functions import *

higgsBosonInpath = "abfss://higgs-boson-container@madbstorage.dfs.core.windows.net/training/"

spark.conf.set("fs.azure.account.key.madbstorage.dfs.core.windows.net", dbutils.secrets.get(scope = "adls-scope", key = "madbstorage-key"))
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")

df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(higgsBosonInpath)

print("Total events: ", df.count())

df.show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vector, DenseVector
from pyspark.sql.types import StructType, StructField, DoubleType

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="Label", outputCol="indexedLabel").fit(df)
transformedDF = labelIndexer.transform(df)

tempDF = transformedDF.drop("Label")

#tempDF.show()

def transform(row):
  tempDict = row.asDict()
  idxLabel = tempDict.pop("indexedLabel", None)
  return (idxLabel, DenseVector(list(tempDict.values())))

higgsFinalDF = tempDF.rdd.map(transform).toDF(["label","features"])

higgsFinalDF.show()

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Split the data into train and test
splits = higgsFinalDF.randomSplit([0.7, 0.3], 1234)
higgsTrainDF = splits[0]
higgsTestDF = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [32, 33, 32, 2]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(higgsTrainDF)

# compute accuracy on the test set
result = model.transform(higgsTestDF)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# COMMAND ----------

import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import azureml
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import pickle

# Verify AML SDK Installed
# view version history at https://pypi.org/project/azureml-sdk/#history 
print("SDK Version:", azureml.core.VERSION)

# Create a Workspace
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "12142336-4e08-46f0-86af-7ec04328e5de"

#Provide values for the Resource Group and Workspace that will be created
resource_group = "higgs-aml-workspace-rg"
workspace_name = "higgs-aml-workspace"
workspace_region = "eastus"

print("Workspace Provisioning complete.")

# COMMAND ----------

# By using the exist_ok param, if the worskpace already exists we get a reference to the existing workspace instead of an error
ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

# COMMAND ----------

# Step 2 - Define a helper method that will use AutoML to train multiple models and pick the best one
##################################################################################################### 
def auto_train_model(ws, experiment_name, full_X, full_Y, training_set_percentage, training_target_accuracy):

    # start a training run by defining an experiment
    experiment = Experiment(ws, experiment_name)
    
    train_X, test_X, train_Y, test_Y = train_test_split(full_X, full_Y, train_size=training_set_percentage, random_state=42)

    train_Y_array = train_Y.values.flatten()

    # Configure the automated ML job
    # The model training is configured to run on the local machine
    # The values for all settings are documented at https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train
    # Notice we no longer have to scale the input values, as Auto ML will try various data scaling approaches automatically
    Automl_config = AutoMLConfig(task = 'classification',
                                 primary_metric = 'accuracy',
                                 max_time_sec = 12000,
                                 iterations = 20,
                                 n_cross_validations = 3,
                                 exit_score = training_target_accuracy,
                                 blacklist_algos = ['kNN','LinearSVM'],
                                 X = train_X,
                                 y = train_Y_array,
                                 path='.\\outputs')

    # Execute the job
    run = experiment.submit(Automl_config, show_output=True)

    # Get the run with the highest accuracy value.
    best_run, best_model = run.get_output()

    return (best_model, run, best_run)

# COMMAND ----------

pdHiggsFinalDF = higgsFinalDF.toPandas()

inputFeatures = pdHiggsFinalDF[["features"][0]]

inputFeatureArray = np.array(inputFeatures.values.tolist())
labelColumn = pdHiggsFinalDF[["label"]]

print(inputFeatureArray)

# COMMAND ----------

# Step 3 - Execute the AutoML driven training
#############################################
experiment_name = "Experiment-AutoML-02"
#model_name = "higgsbosonmodel"
training_set_percentage = 0.70
training_target_accuracy = 0.996
best_model, run, best_run = auto_train_model(ws, experiment_name, inputFeatureArray, labelColumn, training_set_percentage, training_target_accuracy)

# Examine some of the metrics for the best performing run
import pprint
pprint.pprint({k: v for k, v in best_run.get_metrics().items() if isinstance(v, float)})

# COMMAND ----------

# Step 5 - Register the best performing model for later use and deployment
#################################################################
# notice the use of the root run (not best_run) to register the best model
run.register_model(description='AutoML trained higgs boson model')

# COMMAND ----------

# compute accuracy on the test set
result = model.transform(higgsFinalTestDF)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# COMMAND ----------

print("Signal", df.where("Label == 's'").count())
print("Backward", df.where("Label == 'b'").count())

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load training data
data = spark.read.format("libsvm").load("abfss://higgs-boson-container@madbstorage.dfs.core.windows.net/sample-classification-data/")

print(data)

data.show()

# COMMAND ----------

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))