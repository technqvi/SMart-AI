#!/usr/bin/env python
# coding: utf-8

# In[161]:


from google.cloud import bigquery
import pandas as pd
import json

from datetime import date,datetime,timedelta


# In[162]:


table_id = "pongthorn.SMartML.new_incident"
predictResult_table_id="pongthorn.SMartML.new_result_prediction_incident"

mapping_file="incident_sevirity_to_class.json"

PATH_FOLDER_ARTIFACTS="model"
unUsedColtoPredict=['severity','id','severity_id','severity_name','imported_at']


# In[163]:


# with open(mapping_file, 'r') as json_file:
#      map_sevirity_to_class= json.load(json_file)
# print(map_sevirity_to_class)

map_sevirity_to_class={'Cosmatic': 0, 'Minor': 1, 'Major': 2, 'Critical': 3}
print(map_sevirity_to_class)


# In[164]:


# Get today's date
prediction_datetime=datetime.now()

today = date.today()
# Yesterday date
yesterday = today - timedelta(days = 1)
str_today=today.strftime('%Y-%m-%d')
str_yesterday=yesterday.strftime('%Y-%m-%d')
print(f"Get data between {str_yesterday} to {str_today} to predict sevirity level")


# In[165]:


def load_data_bq(sql:str):
 client_bq = bigquery.Client()
 query_result=client_bq.query(sql)
 df=query_result.to_dataframe()
 return df


# In[166]:


sql=f"""
SELECT *  FROM `{table_id}` 
WHERE DATE(imported_at) >= '{str_yesterday}' and DATE(imported_at) < '{str_today}' 

"""
#LIMIT 2
dfNewData=load_data_bq(sql)
dfNewData=dfNewData.drop_duplicates(subset=['id'],keep='first')

dfNewData.insert(2, 'severity', dfNewData['severity_name'].map(map_sevirity_to_class),True)


print(dfNewData.info())
print(dfNewData)


# In[167]:


model = tf.keras.models.load_model(PATH_FOLDER_ARTIFACTS)    
print(f"Load from {PATH_FOLDER_ARTIFACTS}")
print(model.summary())


# In[168]:


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
dfPredictData['prediction_item_date']= datetime.strptime(str(yesterday), '%Y-%m-%d')
dfPredictData['prediction_datetime']=prediction_datetime


# In[169]:


print(dfPredictData.info())
print(dfPredictData)


# In[170]:


#https://cloud.google.com/bigquery/docs/samples/bigquery-create-table#bigquery_create_table-python

try:
    client = bigquery.Client()
    client.get_table(predictResult_table_id)  # Make an API request.
    print("Predict Result Table {} already exists.".format(predictResult_table_id))
except Exception as ex:
    schema = [
    bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("prob_severity", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("predict_severity", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("severity", "INTEGER", mode="REQUIRED"),    
    bigquery.SchemaField("prediction_item_date", "DATETIME", mode="REQUIRED"),    
    bigquery.SchemaField("prediction_datetime", "DATETIME", mode="REQUIRED") 
    ]

    table = bigquery.Table(predictResult_table_id,schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
    type_=bigquery.TimePartitioningType.DAY,field="prediction_item_date")
    
    table = client.create_table(table)  # Make an API request.
    
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )


# In[171]:


def loadDataFrameToBQ():
    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        job = client.load_table_from_dataframe(
            dfPredictData, predictResult_table_id, job_config=job_config
        )
        job.result()  # Wait for the job to complete.
        print("Total Prediction ML ", len(dfPredictData), "Imported igquery successfully")

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




