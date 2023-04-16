# About
We use incident system to perform data analysis and machine learning on a google cloud platform, the main purpose of this project is to analyze data on the incident system in order to develop a machine learning model to predict the severity of each incident with various machine learning algorithms.


## [LoadIncident_PostgresToBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadIncident_PostgresToBQ.ipynb)
- Export incident system data stored on Postgres Database to BigQuery using Bigquery Python.Client Library
- [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt)  , it refer to bigquery table schema and dataframe schema
- Get started with - [Python Client for Google BigQuery](https://cloud.google.com/python/docs/reference/bigquery/latest)

## [QueryIncidentOnBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/QueryIncidentOnBQ.ipynb)
How to retrive from BigQuery by using Python Client for Google BigQuery


## Tutorial#2-2 - [t2-explore_to_create_train_data_ml.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t2-explore_to_create_train_data_ml.ipynb "t2-explore_to_create_train_data_ml.ipynb")
- Retrieve data from BigQuery to prepare data for building Machine Learning.
- Explore & Analyse data with basic statictical method.
- Transform raws data to be more informative and select columns as engineered featured for ML .
- Split Data into Train , Validation and Test Data DataSet and  Ingest them into BigQuery
- [YouTube Tutorial : 2 Exlore Data To Build Traing DataSet For ML](https://www.youtube.com/watch?v=Uzh5Wc4yZSQ)

## Tutorial#2-3 - [t3-build_train_model.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t3-build_train_model.ipynb) | [t3-transform-data.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t3-transform-data.ipynb)
- Get train ,validation and test data from BigQuery as dataframe.
- Covert dataframe to tensorflow as data on tf.data input pipeline
- Normalize numberical value and perform onehot-encoding on Keras Preprocessing Layer.
- Build model with Keras API.
- Train model and evaluate model on validation data and test data.
- Save Model to local path and export it to GCS.
- Reload model to predict data.
- [YouTube Tutorial : 3 Build Tensorflow with Keras API Model To Predict Severity Level of Incident](https://www.youtube.com/watch?v=dplq7B_mp78&t=793s)


## Tutorial#2-4 - [t4-tuning_train_model.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t4-tuning_train_model.ipynb )
Use keras-tuner to find optimal hypter paramter to get best mode ( we improve model performamce from [t3-build_train_model.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t3-build_train_model.ipynb) ).
- [YouTube Tutorial : 4 Tuning Model With Keras Tuner](https://www.youtube.com/watch?v=uDwrhbMMPxw)


## Tutorial#2-5 -[t5-binary_train.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-binary_train.ipynb) | [t5-binary_evaluation.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-binary_evaluation.ipynb) | [t5-evaluate-model-prediction-on-test-data.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-evaluate-model-prediction-on-test-data.ipynb)
- Exlain metrics to evaluatte model focusing on Accuracy.
- Demo how to to evaluate model by getting started with binary classification.
- Demo how to to evaluate model on muticlass classification.
- Describe how balanced data and imbalaned data lead to model's performance
- [YouTube Tutorial : 5 Evaluate Model with Accuracy Metric](https://www.youtube.com/watch?v=itfTFz4e7tg)