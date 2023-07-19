# About
We use incident system to perform data analysis and machine learning on a google cloud platform, the main purpose of this project is to analyze data on the incident system in order to develop a machine learning model to predict the severity of each incident with various machine learning algorithms.


## Main Repo 

### My Tutorial
- [Tutorial on GitHub](https://github.com/technqvi/MyYoutube-Demo/tree/main/google_data_ai)
- [Tutorial on Youtube](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNWPTeGPR5FGj_glwAOuoS7)

### [DailyIncidentForecast](https://github.com/technqvi/SMart-AI/tree/main/DailyIncidentForecast)
Click link to detail.


### [LoadIncident_PostgresToBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadIncident_PostgresToBQ.ipynb)
- Export incident system data stored on Postgres Database to BigQuery using Bigquery Python.Client Library.
- [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt)  , it refer to bigquery table schema and dataframe schema.
- Get started with - [Python Client for Google BigQuery](https://cloud.google.com/python/docs/reference/bigquery/latest).
- Table Schema : [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt) 

### [Model-TF_DF](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF)
Apply [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests/tutorials) to build decision treen model(XGBoost & RandomForest) to predict severity with binary classification (Critical and Noraml Case).

### [QueryIncidentOnBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/QueryIncidentOnBQ.ipynb)
How to retrive from BigQuery by using Python Client for Google BigQuery


### [ExploreToBuildTrainingMLData.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ExploreToBuildTrainingMLData.ipynb)
- Retrieve data from BigQuery to prepare data for building Machine Learning.
- Explore & Analyse data with basic statictical method.
- Transform raws data as engineered featured for buiding ML .
- Split Data into Train , Validation and Test Data DataSet and  Ingest them into BigQuery.


### [Model-TF_Keras](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) (DNN-1-TF-KerasProcessing)
This folder contain folder and file to buid machine learning wiht Tensorflow-Keras, this active folder is DNN-1-TF-KerasProcessing, the others are option. you can go to to  [DNN-1-TF-KerasProcessing](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) to review detail.

- [DNN-1-TF-KerasProcessing](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) (Main Development) : Apply tf.data and  Keras-API and Keras preprocessing layer to tranform data before feeding into  Model. 
- [DNN-2-ScikitLearn](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-2-ScikitLearn)  : tranform data like StandardScaler and OneHot-endcoding with  ScikitLearn.
- [DNN-3-VertextAI-Train](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-3-VertextAI-Train) : transform data manualy on tensforflow dataset.

### [LoadNewIncidentML.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadNewIncidentML.ipynb) | [load-new-incident-ml](https://github.com/technqvi/SMart-AI/tree/main/load-new-incident-ml)
- Script is used to load data from incident table as serving data to get prepared (Excluding all data in training/evaluation/test data) for making prediction.
-  load-new-incident-ml folder is cloud function folder to be ready to deploy.


### [ImportSeverityPrediction_BQ](https://github.com/technqvi/SMart-AI/tree/main/ImportSeverityPrediction_BQ) | [ImportSeverityPredictionToSMApp.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ImportSeverityPredictionToSMApp.ipynb)
- To  import prediction result from new_result_prediction_incident table on Bigquery to Incident System Database, we use Python BigQuery client to do it on the incident application server.
- Prediction Result is shown on Incident Web Site to compare to an actual value determined by Site Manager.
###  [deploy_tf_cloud-func.txt](https://github.com/technqvi/SMart-AI/blob/main/deploy_tf_cloud-func.txt)
Sample shell command to deploy cloud function.
### [DemoDataTransform.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DemoDataTransform.ipynb)
How to apply sklearn.preprocessing to perform normalization for numerical data and one-hot encoding for categorical data.  it is recommended to use 2 ways to get data prepared for training on various machine learning algorithms.



## Addtional Repo 

### [ExportNestedDataOnBigQuery](https://github.com/technqvi/SMart-AI/tree/main/ExportIncidentNestedData)
How to load  nested structure data from PostgresToBQ to BigQuery , the main dsata is incident and nested part is incident detail.
### [Model-BQML](https://github.com/technqvi/SMart-AI/tree/main/Model-BQML)
Script to build ,evaluate and predict severity incidetnt by BigQueryML
### [Model-XGB-RF](https://github.com/technqvi/SMart-AI/tree/main/Model-XGB-RF)
Demo how to build model with XGBoost and  RandomForest



### Tutorial reference : 
- [Train a model using Vertex AI and the Python SDK](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction)
- [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/notebooks/official) (All of them can by applied to real world project)
- [codelabs.developers](https://codelabs.developers.google.com/)
- [introduction_to_tensorflow](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/introduction_to_tensorflow)
- [machine_learning_in_the_enterprise](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/machine_learning_in_the_enterprise/solutions)
- [production_ml](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/production_ml/solutions)
