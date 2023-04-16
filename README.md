# About
We use incident system to perform data analysis and machine learning on a google cloud platform, the main purpose of this project is to analyze data on the incident system in order to develop a machine learning model to predict the severity of each incident with various machine learning algorithms.

### Main reference : 
- [Train a model using Vertex AI and the Python SDK](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction)
- [vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/tree/main/notebooks/official) (All of them can by applied to real world project)
- [codelabs.developers](https://codelabs.developers.google.com/)
- [introduction_to_tensorflow](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/introduction_to_tensorflow)
- [machine_learning_in_the_enterprise](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/machine_learning_in_the_enterprise/solutions)
- [production_ml/](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/machine_learning/deepdive2/production_ml/solutions)


## [LoadIncident_PostgresToBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/LoadIncident_PostgresToBQ.ipynb)
- Export incident system data stored on Postgres Database to BigQuery using Bigquery Python.Client Library.
- [Incident_PostgresToBQ_Schema.txt](https://github.com/technqvi/SMart-AI/blob/main/Incident_PostgresToBQ_Schema.txt)  , it refer to bigquery table schema and dataframe schema.
- Get started with - [Python Client for Google BigQuery](https://cloud.google.com/python/docs/reference/bigquery/latest).

## [QueryIncidentOnBQ.ipynb](https://github.com/technqvi/SMart-AI/blob/main/QueryIncidentOnBQ.ipynb)
How to retrive from BigQuery by using Python Client for Google BigQuery


## [ExploreToBuildTrainingMLData.ipynb](https://github.com/technqvi/SMart-AI/blob/main/ExploreToBuildTrainingMLData.ipynb)
- Retrieve data from BigQuery to prepare data for building Machine Learning.
- Explore & Analyse data with basic statictical method.
- Transform raws data as engineered featured for buiding ML .
- Split Data into Train , Validation and Test Data DataSet and  Ingest them into BigQuery.


## [Model-TF_Keras](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras)
This folder contain folder and file to buid machine learning wiht Tensorflow-Keras, most file apply  , this active folder is DNN-1-TF-KerasProcessing, the others are option. you can go to to  [DNN-1-TF-KerasProcessing](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) to review detail.

- [DNN-1-TF-KerasProcessing](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing) (Main Development) : Apply tf.data and  Keras-API and Keras preprocessing layer to tranform data before feeding into  Model. 
- [DNN-2-ScikitLearn](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-2-ScikitLearn)  : tranform data like StandardScaler and OneHot-endcoding with  ScikitLearn.
- [DNN-3-VertextAI-Train](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-3-VertextAI-Train)


## Tutorial#2-4 - [t4-tuning_train_model.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t4-tuning_train_model.ipynb )
Use keras-tuner to find optimal hypter paramter to get best mode ( we improve model performamce from [t3-build_train_model.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t3-build_train_model.ipynb) ).
- [YouTube Tutorial : 4 Tuning Model With Keras Tuner](https://www.youtube.com/watch?v=uDwrhbMMPxw)


## Tutorial#2-5 -[t5-binary_train.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-binary_train.ipynb) | [t5-binary_evaluation.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-binary_evaluation.ipynb) | [t5-evaluate-model-prediction-on-test-data.ipynb](https://github.com/technqvi/MyYoutube-Demo/blob/main/google_data_ai/t5-evaluate-model-prediction-on-test-data.ipynb)
- Exlain metrics to evaluatte model focusing on Accuracy.
- Demo how to to evaluate model by getting started with binary classification.
- Demo how to to evaluate model on muticlass classification.
- Describe how balanced data and imbalaned data lead to model's performance
- [YouTube Tutorial : 5 Evaluate Model with Accuracy Metric](https://www.youtube.com/watch?v=itfTFz4e7tg)