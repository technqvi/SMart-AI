# About
* This is a pilot project to build a model to make severity-level predictions of incident issues.
* We apply  XGBoost to build model to predict severity level using Decision Forests on Tensorflow Framework .
* It is binary classification , There are 2 classed 1=Critical and 0=Normal.


# Reference
* [Decision Forests on Tensorflow Tutorial](https://www.tensorflow.org/decision_forests/tutorials)
* [Build Severity Incident Model By Keras DNN](https://github.com/technqvi/SMart-AI/tree/main/Model-TF_Keras/DNN-1-TF-KerasProcessing)

### Youtube: [Google Data Analystics & MachineLearning(Part 9-12)](https://www.youtube.com/playlist?list=PLIxgtZc_tZWNWPTeGPR5FGj_glwAOuoS7)

# Main
## [build_incident_ML_v2.ipynb](https://github.com/technqvi/SMart-AI/blob/main/Model-TF_DF/build_incident_ML_v2.ipynb)
* Pull data from Incident DW to create dataset for building model.
* Take raw data from Incident DW to create additional features aside from some initial categorical columns from raw data.
