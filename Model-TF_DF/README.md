# About
* This is a pilot project to build a model to make severity-level predictions of incident issues.
* We apply  XGBoost to build model to predict severity level using Decision Forests on Tensorflow Framework .
* It is binary classification , There are 2 classed 1=Critical and 0=Normal.


# Reference
* [Decision Forests on Tensorflow Tutorial](https://www.tensorflow.org/decision_forests/tutorials)
* [XGBoost With Python By Jason Brownlee](https://machinelearningmastery.com/xgboost-with-python/)
* [Build Severity Incident Model By Keras DNN](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing)

### Youtube: [Building Tensorflow Decision Forests on Google Cloud Data Anlystics & AI](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNpP1Azj4c8kkeTZ3y2gEjl)

# Main
## [build_incident_ML_v2.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/build_incident_ML_v2.ipynb)
* It is used to build the 2 kind of dataset 1.Train/Test dataset for buiding model  2.Unseen dataset(New Data) as inplut for serving model on  production to make prediction 
* Get data from Incident DW to create dataset for building ML model.
* Clean ,transform and enrich data such as creating new feature from some other columns, removing outlier data.
* Import train dataset ,test dataset and unseen dataset into train_incident,test_incident and new_incident tables respectively . 
* [load_v2_new_incident_ml_to_bq](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/load_v2_new_incident_ml_to_bq)  : Google clound function to run this script. 
* Some other tutorials involving this part as links, all of these tutorials are almost same steps as this tutorial : [1 Export Data To BigQuery Using Python](https://studio.youtube.com/video/kgEe4Fb1s1U/edit) | [2 Explore Data To Build Training DataSet For ML](https://studio.youtube.com/video/Uzh5Wc4yZSQ/edit) | [7 Load New Incident To Get Prepared For Making a Prediction](https://studio.youtube.com/video/uR23WkS8XjQ/edit)

  
## [train_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/train_df-tf_incident.ipynb)
## [tune_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/tune_df-tf_incident.ipynb)
## [predict_evaluate_binary_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/predict_evaluate_binary_df-tf_incident.ipynb)

## Option
### [predict_evaluate_multi_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/predict_evaluate_multi_df-tf_incident.ipynb)
### [data](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/data)