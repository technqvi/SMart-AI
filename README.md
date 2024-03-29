# About
* This project involves in building model to predict severity level of incident case  on a google cloud platform. 
* Data for building model and feeding into the model to make prediction has been imported from SMartApp Incident System to BigQuery as DataWarehouse. 
* [SMartApp Web (Django Framework)](https://github.com/technqvi/SMartApp)


## [Severity Prediction Using Gradient Boosted Tree on Tensorflow](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF)
### Tutorial Youtube&GitHub: 
- [Youtube: Building Gradient Boost Tensorflow Model  on Google Data Anlystics & AI](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNpP1Azj4c8kkeTZ3y2gEjl)
- [Youtube:Building Tensorflow Deep Learning Model on Google Data Analystics & Vertext-AI](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNWPTeGPR5FGj_glwAOuoS7)
- [GitHub :Source Code](https://github.com/technqvi/MyYoutube-Demo/tree/main/google_data_ai)

## System Overview  
The process describes step by step aligned to the figure shown in below. Primarily, we write script for each task and schedule it to run Windows scheduler on SMartApp-Server(on-premises) as well as cloud function/cloud scheduler services on google-cloud to execute these tasks.

![OverviewProcess](https://github.com/technqvi/SMart-AI/assets/38780060/80e2ae4c-b65b-4090-9721-1e45b94912b2)


1. Ingest data from Postgres Database that store data of [SMartApp](https://github.com/technqvi/SMartApp) into Incident table on Bigquery
2. Create dataset for developing ML Model from Indident table
   - Train&Test table for building model.
   - Unseen table for serving prediction. 
3. Build model on Train & Test dataset, there are 2 models to serve prediction.  
   - [Decision Tree Model](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF) for Binary classification
   - [Deep Learning Model](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) for Multiclass classification.
4. Export trained model to Google Cloud Storage.
5. Load model from GCS to make prediction to data from Unseen table.
6. Import prediction result  to Prediction Result table.
7. Collect accuracy metric as model performance measurement so that we can monitor how effeciently model perform over time.
8. Postgres Database retrive data from Prediction Result table on BigQuery.
9. Show prediction value on the Incident Update  page in SMartApp Web Site.

### Tool, Framework , Platform & Services
 - Python 3.9 : pandas,numpy,tensorflow,keras tuner,tensorflow decision forests, google-cloud-bigquery
 - Google Cloud Platform: BigQuery,Cloud Storage,Cloud Function,Cloud Schduler, Vertext-AI
 - Application & Datebase : Django Web Framework  and Postgresql

# Build & Tune and Evalute Model 
## [Model-TF_DF](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF) Click link to detail.
Build Gradient Boost Model on Tensorflow Framework to predict severity level such as 1=Critical and 0=Normal.

## [Model-TF_Keras](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) (DNN-1-TF-KerasProcessing)
This folder contain folder and file to buid machine learning wiht Tensorflow-Keras, this active folder is DNN-1-TF-KerasProcessing, the others are option to show different approch  to tranform raw dataset to become proper dataset format for training model. In the part of model design , all of them use the same model,  you can go  to  [DNN-1-TF-KerasProcessing](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) to review detail.

## [DailyIncidentForecast](https://github.com/technqvi/SMart-AI/tree/main/DailyIncidentForecast) Click link to detail.
Build LSTM Time Series Model by taking the number of dialy incident cases over the past 60 days to predict the number incident cases over the next 5 days. 

# Ingest ,Explore and Transform data For Building Dataset

## [LoadIncident_PostgresToBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadIncident_PostgresToBQ.ipynb)
- Export incident system data stored on Postgres Database to BigQuery using Bigquery Python.Client Library.
- [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt)  , it refer to bigquery table schema and dataframe schema.


### [DemoDataTransform.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DemoDataTransform.ipynb)
How to apply sklearn.preprocessing to perform normalization for numerical data and one-hot encoding for categorical data.  it is recommended to use 2 ways to get data prepared for training on various machine learning algorithms.

## [ExploreToBuildTrainingMLData.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ExploreToBuildTrainingMLData.ipynb)
- Retrieve data from BigQuery to prepare data for building Machine Learning.
- Explore & Analyse data with basic statictical method.
- Transform raws data as engineered featured for buiding ML .
- Split Data into Train , Validation and Test Data DataSet and  Ingest them into BigQuery.


## [LoadNewIncidentML.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadNewIncidentML.ipynb) | [load-new-incident-ml](https://github.com/technqvi/SMart-AI/tree/main/load-new-incident-ml)
- Script is used to load data from incident table to build unseen dataset(Excluding all data in training/evaluation/test data)  to feed into  the serving model to  make prediction.
-  load-new-incident-ml folder is cloud function folder to be ready to deploy.

### [BQStream](https://github.com/technqvi/SMart-AI/tree/main/BQStream)
* Demo BigQuery Storage-API for reading and writing stream data on BigQuery. It is new api that make it more efficient to perform ETL process on big data.

# Integrate to Application

##  [ImportSeverityPredictionToSMApp.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ImportSeverityPredictionToSMApp.ipynb)
- We have 2 models to be used to predict severity such as MLP-DNN Model and XGBoost Model.
- To  import prediction result of both  models , we need to do the following tasks.
  - get prediction result from new_result_prediction_incident for multiclassifcation(4 labels such as Cosmetic,Minor,Major and Critical) table on Bigquery
  - retrieve prediction result from new2_result_binary_prediction_incident for binaryclassification(2 labels critical and normal )to Incident System Database
- In practical ,we use Python BigQuery client installed on the incident application server to fullfill these tasks.
- Prediction Result is shown on Incident Web Site to compare to an actual value determined by Site Manager.

###  [deploy_tf_cloud-func.txt](https://github.com/technqvi/SMart-AI/blob/main/deploy_tf_cloud-func.txt)
Sample shell command to deploy cloud function.


# Addtional Repo 


### [Build Model with Other approaches](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras)
- [DNN-2-ScikitLearn](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-2-ScikitLearn)  : tranform data like StandardScaler and OneHot-endcoding with  ScikitLearn.
- [DNN-3-VertextAI-Train](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-3-VertextAI-Train) : transform data manually on tensforflow dataset.

### [QueryIncidentOnBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/QueryIncidentOnBQ.ipynb)
How to retrive from BigQuery by using Python Client for Google BigQuery

### [ExportNestedDataOnBigQuery](https://github.com/technqvi/SMart-AI/tree/main/ExportIncidentNestedData)
How to load  nested structure data as json file from PostgresToBQ to BigQuery , the main data is incident and nested part is incident detail.
### [Model-BQML](https://github.com/technqvi/SMart-AI/tree/main/Model-BQML)
Script to build ,evaluate and predict severity incidetnt by BigQueryML
### [Model-XGB-RF](https://github.com/technqvi/SMart-AI/tree/main/Model-XGB-RF)
Demo how to build model with XGBoost and  RandomForest
* [Use the BigQuery Storage Read API to read table data](https://cloud.google.com/bigquery/docs/reference/storage)
* [Introduction to the BigQuery Storage Write API](https://cloud.google.com/bigquery/docs/write-api)

### [_sample_tf_google](https://github.com/technqvi/SMart-AI/tree/main/_sample_tf_google)
Sample files from [Google Machine Learning Course](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2)

### [Demo_Test](https://github.com/technqvi/SMart-AI/tree/main/Demo_Test)
Grab code from  Tutorial & Sample Reference as link below to test 

### Tutorial & Sample Reference : 
- [Train a model using Vertex AI and the Python SDK](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction)
- [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/notebooks/official) (All of them can by applied to real world project)
- [All Vertex AI code samples](https://cloud.google.com/vertex-ai/docs/samples)
- [Vertex AI Jupyter Notebook tutorials](https://cloud.google.com/vertex-ai/docs/tutorials/jupyter-notebooks)
- [codelabs.developers](https://codelabs.developers.google.com/)
- [introduction_to_tensorflow](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/introduction_to_tensorflow)
- [machine_learning_in_the_enterprise](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/machine_learning_in_the_enterprise)
