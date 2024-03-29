# About
* This project involves building a model to make predictions of severity-level on incident issues.
* We apply  GradientBoosting(XGBoos) that is boosting method of Ensemble Learning Model  to build model to predict severity level using Decision Forests on Tensorflow Framework.
* It is binary classification model, There are 2 classes such as 1=Critical and 0=Normal.
* [Build Severity Incident Model By Keras DNN](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing)

## Tutorial & Demo
### Youtube: [Building Gradient Boost Tensorflow Model  on Google Data Anlystics & AI](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNpP1Azj4c8kkeTZ3y2gEjl)
### Source Code: [demo-tutorial](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/demo-tutorial)

## [Overview Process](https://github.com/technqvi/SMart-AI#system-overview)
Describe how the system work in detail on main repository ,click this link
![image](https://github.com/technqvi/SMart-AI/assets/38780060/b5aba593-4117-4532-9a69-1605f4d0bef8)
 
<img width="549" alt="image" src="https://github.com/technqvi/SMart-AI/assets/38780060/8856a643-f386-4a1b-9cc2-be8832c8cbb1">


## [SMartApp Web (Django Framework)](https://github.com/technqvi/SMartApp)
Click this link

# Main
## [build_incident_ML_v2.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/build_incident_ML_v2.ipynb)
* The steps performed include:
  * It is used to build the 2 kind of datasets 1.Train/Test dataset for buiding model  2.Unseen dataset(New Data) as inplut for the  serving model on  production to make prediction 
  * Get data from Incident DW to create dataset for building ML model.
  * Clean ,transform and enrich data such as creating new feature from some other columns, removing outlier data.
  * Import train dataset ,test dataset and unseen dataset into train_incident,test_incident and new_incident tables respectively . 
* [load_v2_new_incident_ml_to_bq](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/load_v2_new_incident_ml_to_bq)  : Google clound function to run this script. 
* Some other tutorials involving this part as links, all of these tutorials are almost the same steps as this tutorial : [1 Export Data To BigQuery Using Python](https://studio.youtube.com/video/kgEe4Fb1s1U/edit) | [2 Explore Data To Build Training DataSet For ML](https://studio.youtube.com/video/Uzh5Wc4yZSQ/edit) | [7 Load New Incident To Get Prepared For Making a Prediction](https://studio.youtube.com/video/uR23WkS8XjQ/edit)

  
## [train_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/train_df-tf_incident.ipynb)
* Buiding model steps performed include:
  * Load train  and test dataset from BigQuery  as dataframe.
  * Choose features and label(binary classification) from given colums loaded from previous step and transform both to become tensorflow dataset format.
  * Train  model with default hyperparamter on train dataset and Evaluate with test dataset.
  * Inspect and debug model structure and plot chart to show feature importance and model evaluation.
* Tutorial Reference : [Build, train and evaluate models with TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests/tutorials/beginner_colab) | [Inspect and debug decision forest models](https://www.tensorflow.org/decision_forests/tutorials/advanced_colab)
## [tune_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/tune_df-tf_incident.ipynb)
* Tuning model steps performed include:
  * Most steps are similar to [train_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/train_df-tf_incident.ipynb)
  * Tune model with hyper-parameter tuning using TF-DF's tuner and use  automaic optimization by setting use_predefined_hps=True in model contructor.
  * Inspect and debug model structure and training logs.
* [xgb_tf_predict_incident_severity](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/xgb_tf_predict_incident_severity) : Google clound function to run this script. 
* Tutorial Reference : [Automated hyper-parameter tuning](https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab#training_a_model_with_automated_hyper-parameter_tuning_and_automatic_definition_of_the_hyper-parameters_recommended_approach)


## [predict_evaluate_binary_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/predict_evaluate_binary_df-tf_incident.ipynb)
* Evaluation and Prediction model steps performed include:
  * Load unseen data from BigQuery as dataframe and Load model from Google clound storage.
  * Feed unseen data to model to make prediction (1=Critical and 0=Normal).
  * Evaluate model with accuracy metric and use confusion_matrix and classification_report to report model performance.
  * Load prediction result into BigQuery.
* Tutorial Reference : [Making predictions](https://www.tensorflow.org/decision_forests/tutorials/predict_colab)
  

## Option
### [predict_evaluate_multi_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/predict_evaluate_multi_df-tf_incident.ipynb)
This performed the same step as [predict_evaluate_binary_df-tf_incident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/predict_evaluate_binary_df-tf_incident.ipynb),except target class labels that the model will classify label as Multiclass classification (3=Critical,2=Major,1=Minor,0=Cosmetic).
### [data](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_DF/data)
Train and test dataset and unseen dataset as csv files.

# Reference
* [Decision Forests on Tensorflow Tutorial](https://www.tensorflow.org/decision_forests/tutorials)
* [XGBoost With Python By Jason Brownlee](https://machinelearningmastery.com/xgboost-with-python/)