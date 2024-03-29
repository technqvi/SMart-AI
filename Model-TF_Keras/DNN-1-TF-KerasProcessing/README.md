# About
This project involves building multiclass classification and binary classification deep learning models with Tensorflow-Keras to predict severity level on Incident data on BigQuery.


## Main Repo

### [multi_train_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_train_dnn1_incident.ipynb)
- Get train ,validation and test data from BigQuery as dataframe.
- Covert dataframe to tensorflow as data on tf.data input pipeline
- Normalize numberical value and perform onehot-encoding on Keras Preprocessing Layer.
- Build model with Keras API and  Train model.
- Save Model to local path and export it to GCS.
### [multi_tuned_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_tuned_dnn1_incident.ipynb)
- Use BayesianOptimizationon keras-tuner to find optimal hypter paramter to get best model.
- Take best tuned model to retrain to get the N-max epochs.
- Build final model with best model and best  N-max epochs.
- Save Model to local path and export it to GCS.

### [register_deploy_to_explain-ai_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/register_deploy_to_explain-ai_dnn1_incident.ipynb)
Register model and create Endpoint on Vertext-AI and try Vertex Explainable AI feature to gain model explanation to provide better understanding of model prediction result   as like [Introduction to Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)

### [evaluate_score_feature_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/evaluate_score_feature_dnn1_incident.ipynb)
- Explore how balanced the data is on train/evaluation/test.
- Load model from local path to make predction.
- Load Test/Validatoin data and Convert dataframe to tensor format to make predctoin and save result into dataframe.
- Evaluate prediction result on validatoin/test data comapre actual label.
- Register model and create Endpoint on Vertext-AI and try making predction.
- Get a model explanation focusing on feature importance based on feature score that show how much each feature impact to prediction value of both particular sample or whole dataset to gain a better understanding of the model with

### [agg_visualize_feat_imp.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/agg_visualize_feat_imp.ipynb)
* Aggragate  feature score from validation dataset ,test dataset and unseen data  to quantify the importance of each feature 
* Visualize feature importance with bar chart.


### [prediction_serving_tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/prediction_serving_tf_incident.ipynb)
- Get serving data from new_incident table to make prediction.
- Store prediction result into new_result_prediction_incident table.
- [tf1_incident_severity](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/tf1_incident_severity) is cloud function ready to deploy to google cloud converted from [prediction_serving_tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/prediction_serving_tf_incident.ipynb) 



### [bi_train_dnn1_bq_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/bi_train_dnn1_bq_incident.ipynb) | [bi_evaluation_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/bi_evaluation_dnn1_incident.ipynb)
Perform the same things as   [multi_train_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_train_dnn1_incident.ipynb) and  [multi_evaluation_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_evaluation_dnn1_incident.ipynb) respectively. But Use binary classification instead. 



## Addtional Repo

### [incident_sevirity_to_class.json](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/incident_severity_to_class.json) | [incident_sevirity_to_binary.json](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/incident_severity_to_binary.json)
### [demo_model_meta_tf_explanation_metadata.json](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/demo_model_meta_tf_explanation_metadata.json)

### demo_model |  demo_model_with_meta | model | model_binary
These directories store model from training.

###  inoutput 
there are many input instanced format.

### Tools, Services and Framework
- Keras, Keras Tuner and Tensorflow
- Scikit-learn
- Google BigQuery
- Google cloud storage
- Vertext-AI platform
- Cloud function and Cloud scheduler
- Python 3.9 on Anaconda Enviroment






