#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.cloud import bigquery

import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta,timezone

import math
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf  # constantly registered to load model 
print(tf.__version__)
print(tfdf.__version__)


# # Load Configuration Data and Constant Variable

# In[17]:


import functions_framework
@functions_framework.http
def demo_gbt_tf_predict_incident_severity(request):

    all_prediction=False
    is_evaluation=False


    _model="demo_binary_gbt_tf_model"
    model_version=f'{_model}_demo_t150723'

    model_gs_path=f"gs://demo-tf-incident-pongthorn/{_model}"


    projectId="pongthorn"
    dataset_id="SMartML"
    data_table="new2_incident"
    prediction_table="new2_result_binary_prediction_incident"

    unusedCols_unseen=['id','severity_name','imported_at']

    # Get today's date
    prediction_datetime=datetime.now(timezone.utc)
    today_str=prediction_datetime.strftime("%Y-%m-%d")
    today=datetime.strptime(today_str,"%Y-%m-%d")
    print(f"Prediction at {prediction_datetime} for {today_str} ({today})")

    print(model_gs_path)
    print(f"Data: {data_table} and Prediction: {prediction_table}")    


    # # BigQuery Configuration

    # In[18]:


    client = bigquery.Client(project=projectId)
    new_data_table_id=f"{projectId}.{dataset_id}.{data_table}"
    predictResult_table_id=f"{projectId}.{dataset_id}.{prediction_table}"
    print(new_data_table_id)
    print(predictResult_table_id)


    # In[19]:


    try:
        client.get_table(predictResult_table_id)  # Make an API request.
        print("Predict Result Table {} already exists.".format(predictResult_table_id))

    except Exception as ex:
        schema = [
        bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("prediction_item_date", "DATE", mode="REQUIRED"),    
        bigquery.SchemaField("label_binary_severity", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("pred_binary_severity", "INTEGER", mode="REQUIRED"),       
        bigquery.SchemaField("prediction_datetime", "DATETIME", mode="REQUIRED") ,
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED")     
        ]

        table = bigquery.Table(predictResult_table_id,schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,field="prediction_item_date")

        table = client.create_table(table)  # Make an API request.

        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )


    # # Load unseen data(new incident) to Make Prediction

    # In[20]:


    if int(all_prediction)==0:
        sql=f"""
        SELECT *  FROM `{new_data_table_id}` 
         WHERE DATE(imported_at) = '{today_str}'
         order by imported_at
        """
    else:
        sql=f"""
        SELECT *  FROM `{new_data_table_id}` 
         order by imported_at
        """

    print(sql)


    query_result=client.query(sql)
    df=query_result.to_dataframe()
    if df.empty==True:
      print("no data to make prediction")  
      return "no data to make prediction"
    print(df.info())


    # # Build Unseen data by removing label and others

    # In[21]:


    unseen =df.drop(columns=unusedCols_unseen)
    print(unseen.info())
    unseen.tail(5)


    # # Convert dataframe to tensorflow dataset

    # In[22]:


    unseen_ds= tfdf.keras.pd_dataframe_to_tf_dataset(unseen.drop(columns=['severity_id']))
    print(unseen_ds)


    # # Load Model

    # In[23]:


    abc_model = tf.keras.models.load_model(model_gs_path)  
    print(abc_model.summary())


    # # Make prediction

    # In[24]:


    predResultList=abc_model.predict(unseen_ds)
    predServerityIDList=[]
    for predResult in predResultList:
        _class= 1 if predResult[0]>=0.5 else 0  
        predServerityIDList.append(_class) #0=normal , 1=critical
        print(f"{predResult} : {_class}")

    dfPred=pd.DataFrame(data=predServerityIDList,columns=["pred_binary_severity"])      
    dfPred    


    # # Map severity_id to label for actual value.
    # # Merge predicted value to main dataframe

    # In[25]:


    def map_4to2_serverity(severity_id):
        if severity_id==1 or severity_id==2:
           return 1
        else:
           return 0 
    df['label_binary_severity'] =df['severity_id'].apply(map_4to2_serverity)

    dfPred
    df=pd.concat([df,dfPred],axis=1)
    df


    # In[26]:


    # Evaluate model and Show Metric Report


    # In[27]:


    if is_evaluation:
        from sklearn.metrics import confusion_matrix,classification_report
        className=list(set().union(list(df['pred_binary_severity'].unique()),list(df['label_binary_severity'].unique())))
        print(className)
        actualClass=[  f'actual-{x}' for x in  className]
        predictedlClass=[  f'pred-{x}' for x in className]
        y_true=list(df['label_binary_severity'])
        y_pred=list(df['pred_binary_severity'])
        cnf_matrix = confusion_matrix(y_true,y_pred)
        cnf_matrix

        # #index=actual , column=prediction
        cm_df = pd.DataFrame(cnf_matrix,
                             index = actualClass, 
                             columns = predictedlClass)
        print(cm_df)
        print(classification_report(y_true, y_pred, labels=className))


    # # Transform data for Writing Prediction Result to BQ

    # In[18]:


    df=df[['id','label_binary_severity','pred_binary_severity']]
    df['prediction_item_date']=today
    df['prediction_datetime']=datetime.now()
    df['model_version']=model_version
    print(df.tail())


    # # Load data to BQ

    # In[19]:


    def loadDataFrameToBQ():
        # WRITE_TRUNCATE , WRITE_APPEND
        try:
            if all_prediction==1: 
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            else:
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

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


    return f"Imported prediction result  successfully for IsAllPrediction={all_prediction}"

