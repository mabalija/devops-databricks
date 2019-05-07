# Databricks notebook source
# MAGIC %md
# MAGIC Higgs Boson Dataset have the observations captured from the Particle Accelerator to detec whether the newly created particle is a Signal or Background!
# MAGIC ![my_test_image](files/ParticleAccelerator.png)

# COMMAND ----------

from pyspark.sql.functions import *

higgsBosonInpath = "abfss://higgs-boson-container@madbstorage.dfs.core.windows.net/training/"

spark.conf.set("fs.azure.account.key.madbstorage.dfs.core.windows.net", dbutils.secrets.get(scope = "adls-scope", key = "madbstorage-key"))
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")

df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(higgsBosonInpath)

print("Total events2: ", df.count())

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

model.transform(higgsTestDF)

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
resource_group = "aml-workspace-z"
workspace_name = "aml-workspace-z"
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
experiment_name = "Higgs-AutoML-Experiment01"
training_set_percentage = 0.70
training_target_accuracy = 0.996
higgs_best_model, run, best_run = auto_train_model(ws, experiment_name, inputFeatureArray, labelColumn, training_set_percentage, training_target_accuracy)

output_model_path = 'outputs/' + experiment_name + '.pkl'
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
pickle.dump(higgs_best_model,open(output_model_path,'wb'))

# Examine some of the metrics for the best performing run
import pprint
pprint.pprint({k: v for k, v in best_run.get_metrics().items() if isinstance(v, float)})

# COMMAND ----------

# Step 5 - Register the best performing model for later use and deployment
#################################################################
# notice the use of the root run (not best_run) to register the best model
run.register_model(description='AutoML trained higgs boson model')

# COMMAND ----------

from azureml.core.model import Model
registered_model = Model.register(model_path='outputs', model_name=experiment_name + '.pkl', workspace=ws)

# COMMAND ----------

# Step 3 - Download the registered model, re-load  the model and verify it still works
######################################################################################
# Download the model to a local directory
import json
model_path = Model.get_model_path(experiment_name + '.pkl', _workspace=ws)
print('Model download to: ' + model_path)
subArray = inputFeatureArray[0:5]

# Re-load the model
model2 = pickle.load(open(os.path.join(model_path,experiment_name + '.pkl'), 'rb'))

# Use the loaded model to make a prediction
prediction = model2.predict(subArray)
print("8888")
print(model2.__class__)
print("aaa")
print(prediction)
prediction_json = json.dumps(prediction.tolist())
print(prediction_json)



# COMMAND ----------

# Step 4 - Create a Conda dependencies environment file
#######################################################
from azureml.core.conda_dependencies import CondaDependencies 

mycondaenv = CondaDependencies.create(conda_packages=['scikit-learn','numpy','pandas'])

with open("mydeployenv.yml","w") as f:
    f.write(mycondaenv.serialize_to_string())

# COMMAND ----------

#%%writefile score.py
scoring_service = """
import json
import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
import pickle
from sklearn.externals import joblib

def init():
    try:
        # One-time initialization of predictive model and scaler
        from azureml.core.model import Model
        
        global trainedModel   
        global scaler

        model_name = "Higgs-AutoML-Experiment01.pkl" 
        model_path = Model.get_model_path(model_name)
        print('Looking for models in: ', model_path)

        trainedModel = pickle.load(open(os.path.join(model_path,'Higgs-AutoML-Experiment01.pkl'), 'rb'))

    except Exception as e:
        print('Exception during init: ', str(e))

def run(input_json):     
    try:
        inputs = json.loads(input_json)
        
        #Get the scored result
        prediction = json.dumps(trainedModel.predict(inputs).tolist())

    except Exception as e:
        prediction = str(e)
    return prediction
""" 

with open("score.py", "w") as file:
    file.write(scoring_service)

# COMMAND ----------

# Step 5 - Create container image configuration
###############################################
# Build the ContainerImage
runtime = "python" 
driver_file = "score.py"
conda_file = "mydeployenv.yml"

from azureml.core.image import ContainerImage

image_config = ContainerImage.image_configuration(execution_script = driver_file,
                                                  runtime = runtime,
                                                  conda_file = conda_file)

# COMMAND ----------

# Step 6 - Create ACI configuration
####################################
from azureml.core.webservice import AciWebservice, Webservice

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name':'Azure ML ACI'}, 
    description = 'This is a great example.')

# COMMAND ----------

# Step 7 -Deploy the webservice to ACI
######################################
service_name = "higgsautomlexperiment01"

webservice = Webservice.deploy_from_model(
  workspace=ws, 
  name=service_name, 
  deployment_config=aci_config,
  models = [registered_model], 
  image_config=image_config 
  )

webservice.wait_for_deployment(show_output=True)

# COMMAND ----------

print(webservice.location)
print(webservice.description)
print(webservice.get_keys)
print(webservice.scoring_uri)
print(webservice.)

# COMMAND ----------

# Step 11 - Test the AKS deployed webservice
############################################
import json
age = 60
km = 40000
test_data  = json.dumps(inputFeatureArray)
test_data
result = webservice.run(input_data=subArray)
print(result)