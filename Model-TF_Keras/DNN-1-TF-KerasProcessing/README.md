# About
Build multiclass classification and binary classification deep learning model  with Tensorflow-Keras to predict severity level on Incident data on BigQuery, there are serveral steps since building model to serving model to get new data to make prediction.

### Tools, Services and Framework
- Keras, Keras Tuner and Tensorflow
- Scikit-learn
- Google BigQuery
- Google cloud storage
- Vertext-AI platform
- Cloud function and Cloud scheduler
- Python 3.9 on Anaconda Enviroment

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
### [multi_test_evaluation_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_test_evaluation_dnn1_incident.ipynb)
- Explore how balanced the data is on train/evaluation/test.
- Load model from local path to make predction.
- Load Test/Validatoin data and Convert dataframe to tensor format to make predctoin and save result into dataframe.
- Evaluate prediction result on validatoin/test data comapre actual label.
- Register model and create Endpoint on Vertext-AI and try making predction.
### [prediction_serving_tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/prediction_serving_tf_incident.ipynb)
- Get serving data from new_incident table to make prediction.
- Store prediction result into new_result_prediction_incident table.
### [bi_train_dnn1_bq_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/bi_train_dnn1_bq_incident.ipynb) | [bi_test_evaluation_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/bi_test_evaluation_dnn1_incident.ipynb)
Perform the same things as   [multi_train_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_train_dnn1_incident.ipynb) and  [multi_test_evaluation_dnn1_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_Keras/DNN-1-TF-KerasProcessing/multi_test_evaluation_dnn1_incident.ipynb) respectively. But Use binary classication instead. 




## Other Repo
