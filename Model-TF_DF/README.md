# About
* This is a pilot project to build a model to make severity-level predictions of incident issues.
* We apply  XGBoost to build model to predict severity level using Decision Forests on Tensorflow Framework .
* It is binary classification , There are 2 classed 1=Critical and 0=Normal.


# Reference
* [Decision Forests on Tensorflow Tutorial](https://www.tensorflow.org/decision_forests/tutorials)
* [XGBoost With Python By Jason Brownlee](https://machinelearningmastery.com/xgboost-with-python/)
* [Build Severity Incident Model By Keras DNN](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing)

### Youtube: [Google Data Analystics & MachineLearning(Part 9-12)](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNWPTeGPR5FGj_glwAOuoS7)

# Main
## [build_incident_ML_v2.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/build_incident_ML_v2.ipynb)
* It is used to build the 2 kind of dataset 
  * Train/Test dataset for buiding model
  * Unseen dataset(New Data) as inplut for serving model on  production to make prediction 
* Get data from Incident DW to create dataset for building ML model.
* Clean ,transform and enrich data such as creating new feature from some other columns, removing outlier data.
* Import train dataset ,test dataset and unseen dataset into train_incident,test_incident and new_incident tables respectively . 
* See more in other tutorial involving this tutorial as links, all of them are almost same steps as this tutorial : [1 Export Data To BigQuery Using Python](https://studio.youtube.com/video/kgEe4Fb1s1U/edit) | [2 Explore Data To Build Training DataSet For ML](https://studio.youtube.com/video/Uzh5Wc4yZSQ/edit) | [7 Load New Incident To Get Prepared For Making a Prediction](https://studio.youtube.com/video/uR23WkS8XjQ/edit)
  