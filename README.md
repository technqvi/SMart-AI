# About
This project involves in building model to predict severity level of incident case  on a google cloud platform. Mainly, data has been imported from SMartApp Incident System to BigQuery as DataWarehouse. Click to see more [SMartApp-Incident System Project](https://github.com/technqvi/SMartApp)

![OverviewProcess](https://github.com/technqvi/SMart-AI/assets/38780060/80e2ae4c-b65b-4090-9721-1e45b94912b2)

Explain how the system works conceptually aligned with the figure above.
1. Ingest data from Postgres Database that store data of [SMartApp](https://github.com/technqvi/SMartApp) into Incident table on Bigquery
2. Create dataset for developing ML Model from Indident table
   - Train&Test table for building model.
   - Unseen table for serving prediction. 
3. Build model, there are 2 models  
   - [Decision Tree Model](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF) for Binary classification
   - [Deep Learning Model](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) for Multiclass classification.
4. Export trained model to Google Cloud Storage.

### Tool, Framework , Platform & Services
 - Python 3.9 : pandas,numpy,tensorflow,tensorflow decision forests, google-cloud-bigquery
 - Google Cloud Platform : BigQuery,Google Cloud Storage,Google Cloud Function,Google Cloud Schduler, Vertext-AI
 - Application & Datebase : Django Web Framework  and Postgresql

## Main Repo 

### My Tutorial: 
- [Tutorial on Youtube:Building Tensorflow Deep Learning Model on Google Data Analystics & Vertext-AI](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNWPTeGPR5FGj_glwAOuoS7)
- [Tutorial on GitHub : Source Code](https://github.com/technqvi/MyYoutube-Demo/tree/main/google_data_ai)


### [DailyIncidentForecast](https://github.com/technqvi/SMart-AI/tree/main/DailyIncidentForecast) Click link to detail.
Build LSTM Time Series Model by taking the number of dialy incident cases over the past 60 days to predict the number incident cases over the next 5 days. 
### [Model-TF_DF](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF) Click link to detail.
Build XGBoost model to predict severity level such as 1=Critical and 0=Normal  Decision Forests on Tensorflow Framework. 

### [LoadIncident_PostgresToBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadIncident_PostgresToBQ.ipynb)
- Export incident system data stored on Postgres Database to BigQuery using Bigquery Python.Client Library.
- [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt)  , it refer to bigquery table schema and dataframe schema.
- Get started with - [Python Client for Google BigQuery](https://cloud.google.com/python/docs/reference/bigquery/latest).
- Table Schema : [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt) 



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


###  [ImportSeverityPredictionToSMApp.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ImportSeverityPredictionToSMApp.ipynb)
- We have 2 models to be used to predict severity such as MLP-DNN Model and XGBoost Model.
- To  import prediction result of both  models , we need to do the following tasks.
  - get prediction result from new_result_prediction_incident for multiclassifcation(4 labels such as Cosmetic,Minor,Major and Critical) table on Bigquery
  - retrieve prediction result from new2_result_binary_prediction_incident for binaryclassification(2 labels critical and normal )to Incident System Database
- In practical ,we use Python BigQuery client installed on the incident application server to fullfill these tasks.
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
