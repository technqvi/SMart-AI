#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf

import pandas as pd

import json
import os
from datetime import date,datetime,timedelta,timezone

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest

print(tf.__version__)


# In[7]:


import functions_framework
@functions_framework.http
def predict_incident_severity_by_tf(request):


    PATH_FOLDER_ARTIFACTS="gs://tf1-incident-smart-ml-yip/model"
    model_version='model_v2_t150723'
    PROJECT_ID='smart-data-ml'
    dataset_id='SMartML'

    client = bigquery.Client(PROJECT_ID)


    predict_from_date=os.environ.get('predict_from_date', '')
    all_prediction=os.environ.get('all_prediction', '0')  # 1 is all , 0 is 1 day

    print(f"Prediction From = {predict_from_date}")
    print(f"All prediction = {all_prediction}")

    # map_sevirity_to_class={'Cosmatic': 0, 'Minor': 1, 'Major': 2, 'Critical': 3}


    # In[10]:


    table_id = f"{PROJECT_ID}.{dataset_id}.new_incident"
    predictResult_table_id=f"{PROJECT_ID}.{dataset_id}.new_result_prediction_incident"
    unUsedColtoPredict=['severity','id','severity_id','severity_name','imported_at']


    # In[11]:


    print("Load label target multiclasses")
    mapping_file="incident_severity_to_class.json"
    with open(mapping_file, 'r') as json_file:
         map_sevirity_to_class= json.load(json_file)

    print(map_sevirity_to_class)


    # In[12]:


    # Get today's date
    prediction_datetime=datetime.now(timezone.utc)
    today_str=prediction_datetime.strftime("%Y-%m-%d")
    today=datetime.strptime(today_str,"%Y-%m-%d")
    print(prediction_datetime)

    print(f"Prediction at {prediction_datetime} for {today_str} ({today})")


    # In[13]:


    def load_data_bq(sql:str):

     query_result=client.query(sql)
     df=query_result.to_dataframe()
     return df


    # In[14]:


    if int(all_prediction)==0:
        sql=f"""
        SELECT *  FROM `{table_id}` 
         WHERE DATE(imported_at) = '{today_str}' 
         order by imported_at
        """
    else:
        sql=f"""
        SELECT *  FROM `{table_id}` 
         order by imported_at
        """
    print(sql)


    # In[15]:


    dfNewData=load_data_bq(sql)

    if dfNewData.empty==True:
        print("No Data To predict")
        return "No Data To predict"
    else:
        dfNewData=dfNewData.drop_duplicates(subset=['id'],keep='last')
        dfNewData.insert(2, 'severity', dfNewData['severity_name'].map(map_sevirity_to_class),True)
        print(dfNewData)
        print(dfNewData.info())

    try:
        model = tf.keras.models.load_model(PATH_FOLDER_ARTIFACTS)    
        print(f"Load from {PATH_FOLDER_ARTIFACTS}")
        # print(model.summary())
    except Exception as error:

      print(str(error))
      raise error


    # In[18]:


    pdPrediction=pd.DataFrame(columns=['_id','predict_severity','prob_severity'])

    for  row_dict in dfNewData.to_dict(orient="records"):
          incident_id=row_dict['id']
          print(f"{incident_id} - {row_dict['severity']}({row_dict['severity_name']})") 
          for key_removed in unUsedColtoPredict:
           row_dict.pop(key_removed)
          # print(row_dict)  

          input_dict = {name: tf.convert_to_tensor([value]) for name, value in row_dict.items()}


          predictionResult = model.predict(input_dict)
          result_str=','.join([ str(prob) for prob in predictionResult[0]])  
          print(result_str)   

          prob = tf.nn.softmax(predictionResult)
          prob_pct=(100 * prob)  
          _class = tf.argmax(predictionResult,-1).numpy()[0]

          dictPrediction={'_id':incident_id, 'predict_severity':_class,'prob_severity':result_str} 
          pdPrediction =pd.concat([pdPrediction,pd.DataFrame.from_dict([dictPrediction])] )

          print(f"{prob_pct} %   as {_class}")     
          print("======================================================================================")

    dfPredictData=pd.merge(dfNewData,pdPrediction,how='inner',left_on='id',right_on='_id')
    dfPredictData=dfPredictData.drop(columns=['_id'])
    dfPredictData['predict_severity']=dfPredictData['predict_severity'].astype('int')
    dfPredictData=dfPredictData[['id','prob_severity','predict_severity','severity']]
    dfPredictData['prediction_item_date']= datetime.strptime(today_str, '%Y-%m-%d')
    dfPredictData['prediction_datetime']=prediction_datetime
    dfPredictData['model_version']=model_version


    # In[19]:


    print(dfPredictData.info())
    dfPredictData[['id','prediction_item_date','predict_severity','severity']]


    # # Save predictoin resutl to Bigquery

    # In[94]:


    #https://cloud.google.com/bigquery/docs/samples/bigquery-create-table#bigquery_create_table-python

    try:
        client.get_table(predictResult_table_id)  # Make an API request.
        print("Predict Result Table {} already exists.".format(predictResult_table_id))
    except Exception as ex:
        schema = [
        bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("prob_severity", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("predict_severity", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("severity", "INTEGER", mode="REQUIRED"),    
        bigquery.SchemaField("prediction_item_date", "DATETIME", mode="REQUIRED"),    
        bigquery.SchemaField("prediction_datetime", "DATETIME", mode="REQUIRED"), 
        bigquery.SchemaField("model_version",  "STRING", mode="REQUIRED")     
        ]

        table = bigquery.Table(predictResult_table_id,schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,field="prediction_item_date")

        table = client.create_table(table)  # Make an API request.

        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )


    # In[201]:


    def loadDataFrameToBQ():
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
            )

            job = client.load_table_from_dataframe(
                dfPredictData, predictResult_table_id, job_config=job_config
            )
            job.result()  # Wait for the job to complete.
            print("Total Prediction ML ", len(dfPredictData), "Imported bigquery successfully")

        except BadRequest as e:
            print("Bigquery Error\n")
            for e in job.errors:
                print('ERROR: {}'.format(e['message']))

    try:
        loadDataFrameToBQ()
    except Exception as ex:
        raise ex


    # In[ ]:





    # In[ ]:





    # In[ ]:


    return 'All incidents has been predicted completely.'







# In[ ]:




