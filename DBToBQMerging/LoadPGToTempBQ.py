#!/usr/bin/env python
# coding: utf-8

# # Imported Library

# In[108]:


import psycopg2
from psycopg2 import sql
import psycopg2.extras as extras
import pandas as pd
import json
from datetime import datetime,timezone
from dateutil import tz

import os
import sys 

from configupdater import ConfigUpdater
# pip install ConfigUpdater

from dotenv import dotenv_values

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest
from google.oauth2 import service_account



# In[109]:


is_py=True
view_name = "pmr_pm_plan"
isFirstLoad=False
if is_py:
    press_Y=''
    ok=False

    if len(sys.argv) > 1:
        view_name=sys.argv[1]
    else:
        print("Enter the following input: ")
        view_name = input("View Table Name : ")
print(f"View name to load to BQ :{view_name}")


# # Imported date

# In[110]:


dt_imported=datetime.now(timezone.utc) # utc
dt_imported=datetime.strptime(dt_imported.strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
print(f"UTC: {dt_imported}")



# # Set view

# In[111]:


log = "models_logging_change"
if view_name == "pmr_pm_plan":
    content_id = 36
    view_name_id = "pm_id"

elif view_name == "pmr_pm_item":
    content_id = 37
    view_name_id = "pm_item_id"

elif view_name == "pmr_project":
    content_id = 7
    view_name_id = "project_id"

elif view_name == "pmr_inventory":
    content_id = 14
    view_name_id = "inventory_id"

else:
    raise Exception("No specified content type id")


# # Set data and cofig path

# In[112]:


projectId='smart-data-ml'  # smart-data-ml  or kku-intern-dataai
dataset_id='SMartData_Temp'  # 'SMartData_Temp'  'PMReport_Temp'
main_dataset_id='SMartDataAnalytics'  # ='SMartDataAnalytics'  'PMReport_Main'
credential_file=r"C:\Windows\smart-data-ml-91b6f6204773.json"  
# C:\Windows\smart-data-ml-91b6f6204773.json
# C:\Windows\kku-intern-dataai-a5449aee8483.json




# In[113]:


credentials = service_account.Credentials.from_service_account_file(credential_file)

table_name=view_name.replace("pmr_","temp_") #can change in ("name") to temp table
table_id = f"{projectId}.{dataset_id}.{table_name}"
print(table_id)


main_table_name=view_name.replace("pmr_","")
main_table_id = f"{projectId}.{main_dataset_id}.{main_table_name}"
print(main_table_id)

# https://cloud.google.com/bigquery/docs/reference/rest/v2/Job
to_bq_mode="WRITE_EMPTY"


client = bigquery.Client(credentials= credentials,project=projectId)


# Read Configuration File and Initialize BQ Object

# In[114]:


updater = ConfigUpdater()
updater.read(".cfg")

env_path='.env'
config = dotenv_values(dotenv_path=env_path)


# In[115]:


last_imported=datetime.strptime(updater["metadata"][view_name].value,"%Y-%m-%d %H:%M:%S")
print(f"UTC:{last_imported}")

# local_zone = tz.tzlocal()
# last_imported = last_imported.astimezone(local_zone)
# print(f"Local Asia/Bangkok:{last_imported}")


# # Postgres &BigQuery

# In[116]:


def get_postgres_conn():
 try:
  conn = psycopg2.connect(
        database=config['DATABASES_NAME'], user=config['DATABASES_USER'],
      password=config['DATABASES_PASSWORD'], host=config['DATABASES_HOST']
     )
  return conn

 except Exception as error:
  print(error)      
  raise error
def list_data(sql,params,connection):
 df=None   
 with connection.cursor() as cursor:
    
    if params is None:
       cursor.execute(sql)
    else:
       cursor.execute(sql,params)
    
    columns = [col[0] for col in cursor.description]
    dataList = [dict(zip(columns, row)) for row in cursor.fetchall()]
    df = pd.DataFrame(data=dataList) 
 return df 


# In[117]:


def get_bq_table():
 try:
    table=client.get_table(table_id)  # Make an API request.
    print("Table {} already exists.".format(table_id))
    print(table.schema)
    return True
 except NotFound:
    raise Exception("Table {} is not found.".format(table_id))
    
def collectBQError(x_job):
 if x_job.errors is not None:
    for error in x_job.errors:  
      msg=f"{error['reason']} - {error['message']}"
      listError.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dtStr_imported,source_name,msg])
    if   len(listError)>0:
     logErrorMessage(listError,False)  

    
def insertDataFrameToBQ(df_trasns):
    try:
        job_config = bigquery.LoadJobConfig(write_disposition=to_bq_mode,)
        job = client.load_table_from_dataframe(df_trasns, table_id, job_config=job_config)
        try:
         job.result()  # Wait for the job to complete.
        except ClientError as e:
         print(job.errors)

        print("Total ", len(df_trasns), f"Imported data to {table_id} on bigquery successfully")

    except BadRequest as e:
        print("Bigquery Error\n")
        print(e) 


# # Check whether it is the first loading?

# In[120]:


print("If the main table is empty , so the action of each row  must be 'added' on temp table")
rows_iter   = client.list_rows(main_table_id, max_results=1) 
no_main=len(list(rows_iter))
if no_main==0:
 isFirstLoad=True
 print(f"This is the first loaing , so there is No DATA in {main_table_id}, we load all rows from {view_name} to import into {table_id} action will be 'added' ")


# # For The next Load
# * get data from model log based on condition last_imported and table
# * Get all actions from log table by selecting unique object_id and setting by doing something as logic
# * Create  id and action dataframe form filtered rows from log table

# In[121]:


def list_model_log(x_last_imported,x_content_id):
    sql_log = f"""
    SELECT object_id, action,TO_CHAR(date_created,'YYYY-MM-DD HH24:MI:SS') as date_created FROM {log}
    WHERE date_created  AT time zone 'utc' >= '{x_last_imported}' AND content_type_id = {x_content_id} ORDER BY object_id, date_created
    """
    print(sql_log)


    # Asia/Bangkok 
    lf = list_data(sql_log, None, get_postgres_conn())
    print(f"Retrieve all rows after {last_imported}")
    print(lf.info())
    return lf


# In[122]:


def select_actual_action(lf):
    listIDs=lf["object_id"].unique().tolist()
    listUpdateData=[]
    for id in listIDs:
        lfTemp=lf.query("object_id==@id")
        # print(lfTemp)
        # print("----------------------------------------------------------------")


        first_row = lfTemp.iloc[0]
        last_row = lfTemp.iloc[-1]
        # print(first_row)
        # print(last_row)

        if len(lfTemp)==1:
            listUpdateData.append([id,first_row["action"]])
        else:
            if first_row["action"] == "added" and last_row["action"] == "deleted":
                continue
            elif first_row["action"] == "added" and last_row["action"] != "deleted":
                listUpdateData.append([id,"added"])
            else : listUpdateData.append([id,last_row["action"]])

    print("Convert listUpdate to dataframe")
    dfUpdateData = pd.DataFrame(listUpdateData, columns= ['id', 'action'])
    dfUpdateData['id'] = dfUpdateData['id'].astype('int64')
    dfUpdateData=dfUpdateData.sort_values(by="id")
    dfUpdateData=dfUpdateData.reset_index(drop=True)

    return dfUpdateData


# In[93]:


if isFirstLoad==False:
    dfModelLog=list_model_log(last_imported,content_id)
    if dfModelLog.empty==True:
        print("No row to be imported.")
        exit()
    else:
       dfModelLog=select_actual_action( dfModelLog)
       listModelLogObjectIDs=dfModelLog['id'].tolist()
       print(dfModelLog.info())
       print(dfModelLog)       
       print(listModelLogObjectIDs) 


# # Load view and transform

# In[94]:


if isFirstLoad==False:
    if len(listModelLogObjectIDs)>1:
     sql_view=f"select *  from {view_name}  where {view_name_id} in {tuple(listModelLogObjectIDs)}"
    else:
     sql_view=f"select *  from {view_name}  where {view_name_id} ={listModelLogObjectIDs[0]}"
else:
     sql_view=f"select *  from {view_name}  where  updated_at AT time zone 'utc' >= '{last_imported}'"
        

print(sql_view)
df=list_data(sql_view,None,get_postgres_conn())


if df.empty==True:
    print("No row to be imported.")
    exit()

df=df.drop(columns='updated_at')


if isFirstLoad:
    df['action']='added'
    
print(df.info())
df


# # Data Transaformation
# * IF The first load then add actio='Added'
# * IF The nextload then Merge LogDF and ViewDF and add deleted row 
#   * Get Deleted Items  to Create deleted dataframe by using listDeleted
#   * If there is one deletd row then  we will merge it to master dataframe

# In[95]:


def add_acutal_action_to_df_at_next(df,dfUpdateData):
    merged_df = pd.merge(df, dfUpdateData, left_on=view_name_id, right_on='id', how='inner')
    merged_df = merged_df.drop(columns=['id'])

    listSelected = df[view_name_id].tolist()
    print(listSelected)

    set1 = set(listModelLogObjectIDs)
    set2 = set(listSelected)
    listDeleted = list(set1.symmetric_difference(set2))

    print(listDeleted)

    if len(listDeleted)>0:
        print("There are some deleted rows")
        dfDeleted=pd.DataFrame(data=listDeleted,columns=[view_name_id])
        dfDeleted['action']='deleted'
        print(dfDeleted)
        merged_df=pd.concat([merged_df,dfDeleted],axis=0)

    else:
        print("No row deleted")

    return merged_df    


# # Check duplicate ID

# In[96]:


if isFirstLoad==False:
 df=add_acutal_action_to_df_at_next(df,dfModelLog)
 

# merged_df['imported_at']=dt_imported
df=df.reset_index(drop=True  )
print(df.info())
print(df)


# In[97]:


hasDplicateIDs = df[view_name_id].duplicated().any()
if  hasDplicateIDs:
 raise Exception("There are some duplicate id on dfUpdateData")
else:
 print(f"There is no duplicate {view_name_id} ID")  


# # Insert data to BQ data frame

# In[98]:


if get_bq_table():
    try:
        insertDataFrameToBQ(df)
    except Exception as ex:
        raise ex


# In[99]:


updater["metadata"][view_name].value=dt_imported.strftime("%Y-%m-%d %H:%M:%S")
updater.update_file() 


# In[100]:


print(datetime.now(timezone.utc) )


# In[ ]:





# In[ ]:





# In[ ]:




