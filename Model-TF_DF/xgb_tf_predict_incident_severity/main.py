#!/usr/bin/env python
# coding: utf-8

# In[25]:

from google.cloud import bigquery

import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta

import math
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf  # constantly registered to load model 
print(tf.__version__)
print(tfdf.__version__)


# In[ ]:


import functions_framework
@functions_framework.http
def xgb_tf_predict_incident_severity(request):


    # In[26]:


    model_tree_type=1 # 1= xgboost  2-random forest
    option=1

    if model_tree_type==1:
        _model='model_xgb_tf'
    else:
         _model='model_rf_tf'

    model_gs_path=f"gs://demo-tuned-tf-incident-pongthorn/{_model}"
    print(model_gs_path)

    projectId="pongthorn"
    dataset_id="SMartML"

    map_severity_to_class={0:4,1: 3, 2: 2, 3: 1}

    if option==1:
        unusedCols_unseen=['id','severity_name','open_to_close_hour']
    else:
        unusedCols_unseen=['id','severity_name','range_open_to_close_hour']



    # # Load Model

    # In[27]:


    abc_model = tf.keras.models.load_model(model_gs_path)  
    print(abc_model.summary())
    # abc_model = tf.keras.models.load_model(model_local_path) 


    # # Make Prediction

    # In[28]:


    client = bigquery.Client(project=projectId)
    new_data_table_id=f"{projectId}.{dataset_id}.new2_incident"

    query_result=client.query(f"SELECT * FROM {new_data_table_id}")
    df=query_result.to_dataframe()

    unseen =df.drop(columns=unusedCols_unseen)
    print(unseen.info())
    unseen.tail(10)


    # In[29]:


    unseen_ds= tfdf.keras.pd_dataframe_to_tf_dataset(unseen.drop(columns=['severity_id']))
    unseen_ds


    # In[30]:


    predResultList=abc_model.predict(unseen_ds)
    predServerityIDList=[]
    for predResult in predResultList:
        _class=tf.argmax(predResult,-1).numpy()
        pred_seveirty_id=map_severity_to_class[_class]
        predServerityIDList.append(pred_seveirty_id)
        print(f"{predResult} : {_class} as severity#{pred_seveirty_id}")

    dfPred=pd.DataFrame(data=predServerityIDList,columns=["pred_severity_id"])   
    df=pd.concat([dfPred,df],axis=1)
    # df


    # # Evaluate model

    # In[31]:


    # from sklearn.metrics import confusion_matrix,classification_report
    # className=list(set().union(list(df['pred_severity_id'].unique()),list(df['severity_id'].unique())))
    # className
    # actualClass=[  f'actual-{x}' for x in  className]
    # predictedlClass=[  f'pred-{x}' for x in className]
    # y_true=list(df['severity_id'])
    # y_pred=list(df['pred_severity_id'])
    # cnf_matrix = confusion_matrix(y_true,y_pred)
    # cnf_matrix

    # # #index=actual , column=prediction
    # cm_df = pd.DataFrame(cnf_matrix,
    #                      index = actualClass, 
    #                      columns = predictedlClass)
    # print(cm_df)
    # print(classification_report(y_true, y_pred, labels=className))


    # # Write Prediction Result to BQ

    # In[32]:


    df=df[['id','severity_id','pred_severity_id']]
    df['prediction_datetime']=datetime.now()
    print(df)


    # In[ ]:





    # In[33]:


    predictResult_table_id=f"{projectId}.{dataset_id}.new2_result_prediction_incident"

    try:
        client.get_table(predictResult_table_id)  # Make an API request.
        print("Predict Result Table {} already exists.".format(predictResult_table_id))
    except Exception as ex:
        schema = [
        bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("severity_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("pred_severity_id", "INTEGER", mode="REQUIRED"),       
        bigquery.SchemaField("prediction_datetime", "DATETIME", mode="REQUIRED") 
        ]

        table = bigquery.Table(predictResult_table_id,schema=schema)
        # table.time_partitioning = bigquery.TimePartitioning(
        # type_=bigquery.TimePartitioningType.DAY,field="prediction_item_date")

        table = client.create_table(table)  # Make an API request.

        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )


    # In[34]:


    def loadDataFrameToBQ():
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
            )

            job = client.load_table_from_dataframe(
                df, predictResult_table_id, job_config=job_config
            )
            job.result()  # Wait for the job to complete.
            print("Total Prediction ML ", len(df), "Imported bigquery successfully")

        except BadRequest as e:
            print("Bigquery Error\n")
            for e in job.errors:
                print('ERROR: {}'.format(e['message']))

    try:
        loadDataFrameToBQ()
    except Exception as ex:
        raise ex


    # In[ ]:


    return 'All incidents has been predicted completely.'

