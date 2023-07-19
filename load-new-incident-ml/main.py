#!/usr/bin/env python
# coding: utf-8

# In[22]:


from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime ,timezone
from google.oauth2 import service_account

import functions_framework

from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest
import os


# In[23]:


@functions_framework.http
def load_new_incident_ml_to_bq(request):


    # In[24]:


    projectId='smart-data-ml'

    start_date_query=os.environ.get('start_date_query', '2023-07-16')
    start_date_query='2023-07-16'


    # In[25]:


    table_ml_id = f"{projectId}.SMartML.new_incident"
    table_dw_id=f"{projectId}.SMartDW.incident"

    # credentials = service_account.Credentials.from_service_account_file(r'C:\Windows\smart-data-ml-91b6f6204773.json')
    # client = bigquery.Client(credentials=credentials, project=projectId)

    client = bigquery.Client(project=projectId)


    # In[26]:


    # Get Last Upldate from BQ update data
    dateCols=['open_datetime','close_datetime','response_datetime','resolved_datetime']

    removeCols=dateCols+['open_to_close','response_to_resolved']

    numbericCols=['open_to_close_hour','response_to_resolved_hour']
    cateCols=['sla','product_type','brand','service_type','incident_type']


    # In[27]:


    #https://cloud.google.com/bigquery/docs/samples/bigquery-create-table#bigquery_create_table-python

    try:
        client.get_table(table_ml_id)  # Make an API request.
        print("Table {} already exists.".format(table_ml_id))
    except Exception as ex:
        schema = [
        bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("severity_id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("severity_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("sla", "STRING", mode="REQUIRED"),    
        bigquery.SchemaField("product_type", "STRING", mode="REQUIRED"),  
        bigquery.SchemaField("brand", "STRING", mode="REQUIRED"),  
        bigquery.SchemaField("service_type", "STRING", mode="REQUIRED"),  
        bigquery.SchemaField("incident_type", "STRING", mode="REQUIRED"),  
        bigquery.SchemaField("open_to_close_hour", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("response_to_resolved_hour", "FLOAT", mode="REQUIRED"),    
        bigquery.SchemaField("imported_at", "DATETIME", mode="REQUIRED")    
        ]

        table = bigquery.Table(table_ml_id,schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,field="imported_at")


        table = client.create_table(table)  # Make an API request.
        print(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )


    # In[28]:


    dt_imported=datetime.now(timezone.utc)
    str_imported=dt_imported.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Imported DateTime: {str_imported}" )

    sql_lastImport=f"SELECT max(imported_at) as last_imported from `{table_ml_id}` "

    print(sql_lastImport)

    job_lastImported=client.query(sql_lastImport)
    str_lastImported=None
    for row in job_lastImported:    
        if row.last_imported is not None: 
            str_lastImported=row.last_imported.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Last Imported DateTime: {str_lastImported}" )

    if str_lastImported is not None:
      print("Start date from last loading")  
      start_date_query=str_lastImported
    else:
      print("Init First loading")  


    print(f"Start Import on update_at of last imported date : {start_date_query}" )


    # In[29]:


    sql=f"""
    SELECT  id,
    severity_id,severity_name,sla,
    product_type,brand,service_type,incident_type,
    open_datetime,  close_datetime, response_datetime,resolved_datetime
    FROM `{table_dw_id}` 
    WHERE imported_at>='{start_date_query}'
    order by imported_at
    """
    #WHERE imported_at>='{start_date_query}' and imported_at<='2023-03-24'
    #WHERE imported_at>='{start_date_query}'

    print(sql)

    query_result=client.query(sql)
    df_all=query_result.to_dataframe()
    df_all=df_all.drop_duplicates(subset=['id'],keep='last')
    print(df_all.info())
    df_all.head()


    # In[30]:


    if len(df_all)==0:
     print("No record to load")   
     # return "No record to load"  


    # In[31]:


    start_end_list=[ ['open_datetime','close_datetime'],['response_datetime','resolved_datetime']]
    listDiffDateDeltaCols=[]
    listDiffHourCols=[]
    for item  in  start_end_list:
       diff_str=f"{item[0]}_to_{item[1]}" 
       diff_str=diff_str.replace('_datetime','')  
       listDiffDateDeltaCols.append(diff_str)
       df_all[diff_str]=df_all[item[1]]-df_all[item[0]]

       diff_hour=f'{diff_str}_hour'
       listDiffHourCols.append(diff_hour)
       df_all[diff_hour] = df_all[diff_str].apply(lambda x:  x.total_seconds() / (60*60) if x is not np.nan else np.nan  )


    # In[32]:


    for col in numbericCols:
     df_all=df_all.query(f'{col}!=0')

    # get only last update of id
    df_all=df_all.drop_duplicates(subset=['id'],keep='first')
    df_all=df_all.drop(columns=removeCols)

    df_all['imported_at']=dt_imported

    df_all.dropna(inplace=True)

    print(df_all.info())
    print(df_all.tail())


    # In[19]:


    # print(df_all[numbericCols].describe(percentiles=[.9,.75,.50,.25,.10]))


    # In[20]:


    # df_all.to_csv("data/New_Incident.csv",index=False)


    # In[ ]:





    # In[21]:


    def loadDataFrameToBQ():
        try:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
            )

            job = client.load_table_from_dataframe(
                df_all, table_ml_id, job_config=job_config
            )
            job.result()  # Wait for the job to complete.
            print("Total ", len(df_all), "Imported bigquery successfully")

        except BadRequest as e:
            print("Bigquery Error\n")
            for e in job.errors:
                print('ERROR: {}'.format(e['message']))

    try:
        loadDataFrameToBQ()
    except Exception as ex:
        raise ex


    # In[ ]:


    return 'ok'


# In[ ]:


# if __name__ == "__main__":
#  result=load_new_incident_ml_to_bq(None)
#  print(result)

