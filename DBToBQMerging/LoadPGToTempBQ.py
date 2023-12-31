#!/usr/bin/env python
# coding: utf-8

# # Imported Library

# In[89]:


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

# [metadata]
# pmr_pm_plan = 2019-01-01 00:00:00
# pmr_pm_item = 2019-01-01 00:00:00
# pmr_project = 2019-01-01 00:00:00
# pmr_inventory  = 2019-01-01 00:00:00


# In[90]:


is_py=True
view_name = ""
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

# In[91]:


dt_imported=datetime.now(timezone.utc) # utc
dt_imported=datetime.strptime(dt_imported.strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S")
print(f"UTC: {dt_imported} For This Import")



# # Set view data and log table

# In[92]:


log = "models_logging_change"
def get_contentID_keyName(view_name):

    if view_name == "pmr_pm_plan":
        tableContentID = 36
        key_name = "pm_id"
        sp="merge_pm_plan"

    elif view_name == "pmr_pm_item":
        tableContentID = 37
        key_name = "pm_item_id"
        sp="merge_pm_item"

    elif view_name == "pmr_project":
        tableContentID = 7
        key_name = "project_id"
        sp="merge_project"

    elif view_name == "pmr_inventory":
        tableContentID = 14
        key_name = "inventory_id"
        sp="merge_inventory"

    else:
        raise Exception("No specified content type id")
        
    return tableContentID, key_name,sp


content_id , view_name_id,sp_name=get_contentID_keyName(view_name)
print(content_id," - ",view_name_id," - ",sp_name)


# # Set data and cofig path

# In[93]:


# Test config,env file and key to be used ,all of used key  are existing.
cfg_path="cfg_last_import"
env_path='.env'

updater = ConfigUpdater()
updater.read(os.path.join(cfg_path,f"{view_name}.cfg"))

config = dotenv_values(dotenv_path=env_path)

print(env_path)
print(cfg_path)


# In[94]:


# Test exsitng project dataset and table anme

projectId=config['PROJECT_ID']  # smart-data-ml  or kku-intern-dataai or ponthorn
credential_file=config['PROJECT_CREDENTIAL_FILE']
# C:\Windows\smart-data-ml-91b6f6204773.json
# C:\Windows\kku-intern-dataai-a5449aee8483.json
# C:\Windows\pongthorn-5decdc5124f5.json


dataset_id='SMartData_Temp'  # 'SMartData_Temp'  'PMReport_Temp'
main_dataset_id='SMartDataAnalytics'  # ='SMartDataAnalytics'  'PMReport_Main'

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

# In[95]:


last_imported=datetime.strptime(updater["metadata"][view_name].value,"%Y-%m-%d %H:%M:%S")
print(f"UTC:{last_imported}  Of Last Import")

# local_zone = tz.tzlocal()
# last_imported = last_imported.astimezone(local_zone)
# print(f"Local Asia/Bangkok:{last_imported}")


# # Postgres &BigQuery

# In[96]:


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


# In[97]:


def get_bq_table():
 try:
    table=client.get_table(table_id)  # Make an API request.
    print("Table {} already exists.".format(table_id))
    print(table.schema)
    return True
 except NotFound:
    raise Exception("Table {} is not found.".format(table_id))
    

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

# In[98]:


def checkFirstLoad():
    print("If the main table is empty , so the action of each row  must be 'added' on temp table")
    rows_iter   = client.list_rows(main_table_id, max_results=1) 
    no_main=len(list(rows_iter))
    if no_main==0:
     isFirstLoad=True
     print(f"This is the first loaing , so there is No DATA in {main_table_id}, we load all rows from {view_name} to import into {table_id} action will be 'added' ")
    else:
     isFirstLoad=False   
    return isFirstLoad


# In[99]:


isFirstLoad=checkFirstLoad()
print(f"IsFirstLoad={isFirstLoad}")


# # For The next Load
# * get data from model log based on condition last_imported and table
# * Get all actions from log table by selecting unique object_id and setting by doing something as logic
# * Create  id and action dataframe form filtered rows from log table

# In[100]:


def list_model_log(x_last_imported,x_content_id):
    sql_log = f"""
    SELECT object_id, action,TO_CHAR(date_created,'YYYY-MM-DD HH24:MI:SS') as date_created ,changed_data
    FROM {log}
    WHERE date_created  AT time zone 'utc' >= '{x_last_imported}' AND content_type_id = {x_content_id} 
    ORDER BY object_id, date_created
    """
    print(sql_log)


    # Asia/Bangkok 
    lf = list_data(sql_log, None, get_postgres_conn())
    print(f"Retrieve all rows after {last_imported}")
    print(lf.info())
    return lf


# In[101]:


def check_any_changes_to_collumns_view(dfAction,x_view_name,_x_key_name):
    """
    Check dataframe from log model that contain only changed action to select changed fields on view.
    """

    listACtion=dfAction["action"].unique().tolist()
    if len(listACtion)==1 and listACtion[0]=='changed':
        print("###########################################################")
        print("Process dataframe containing only all changed action")
        print(dfAction)
        print("###########################################################")
    
    


# In[102]:


def select_actual_action(lf):
    listIDs=lf["object_id"].unique().tolist()
    listUpdateData=[]
    for id in listIDs:
        lfTemp=lf.query("object_id==@id")
        print(lfTemp)
        print("----------------------------------------------------------------")
        
        
        # check_any_changes_to_collumns_view(lfTemp,content_id,view_name_id)


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


# In[103]:


if isFirstLoad==False:
    listModelLogObjectIDs=[]
    dfModelLog=list_model_log(last_imported,content_id)
    if dfModelLog.empty==True:
        print("No row to be imported.")
        exit()
    else:
       print("Get row imported from model log to set action") 
       dfModelLog=select_actual_action( dfModelLog)
       listModelLogObjectIDs=dfModelLog['id'].tolist()
       print(dfModelLog.info())
       print(dfModelLog)       
       print(listModelLogObjectIDs) 


# # Load view and transform

# In[104]:


def retrive_next_data_from_view(x_view,x_id,x_listModelLogObjectIDs):
    if len(x_listModelLogObjectIDs)>1:
     sql_view=f"select *  from {x_view}  where {x_id} in {tuple(x_listModelLogObjectIDs)}"
    else:
     sql_view=f"select *  from {x_view}  where {x_id} ={x_listModelLogObjectIDs[0]}"
    
    print(sql_view)
    df=list_data(sql_view,None,get_postgres_conn())

    if df.empty==True:
     return df
    df=df.drop(columns='updated_at')
    return df 


def retrive_first_data_from_view(x_view,x_last_imported):
     sql_view=f"select *  from {x_view}  where  updated_at AT time zone 'utc' >= '{x_last_imported}'"
     print(sql_view)
     df=list_data(sql_view,None,get_postgres_conn())
     if df.empty==True:
            return df
     df=df.drop(columns='updated_at')
     df['action']='added'
     return df   
def retrive_one_row_from_view_to_gen_df_schema(x_view):
    sql_view=f"select *  from {x_view}  limit 1"
    print(sql_view)
    df=list_data(sql_view,None,get_postgres_conn())
    df=df.drop(columns='updated_at')
    return df


if isFirstLoad:
 df=retrive_first_data_from_view(view_name,last_imported)
 if df.empty==True:
    print("No row to be imported.")
    exit()
 else:
    print(df.info())
else:
 df=retrive_next_data_from_view(view_name,view_name_id,listModelLogObjectIDs)  
 if df.empty==True:
    print("Due to having deleted items, we will Get schema from {} to create empty dataframe with schema.")
    df=retrive_one_row_from_view_to_gen_df_schema(view_name)
    # this id has been included in listModelLogObjectIDs which contain deleted action , so we can use it as schema generation
    print(df)

    
    
    


# # Data Transaformation
# * IF The first load then add actio='Added'
# * IF The nextload then Merge LogDF and ViewDF and add deleted row 
#   * Get Deleted Items  to Create deleted dataframe by using listDeleted
#   * If there is one deletd row then  we will merge it to master dataframe
# * IF the next load has only deleted action

# In[105]:


def add_acutal_action_to_df_at_next(df,dfUpdateData,x_view,x_id):
    # merget model log(id and action) to data view
    # if  dfUpdateData  contain only deleted action
    # we will merge to get datafdame shcema, it can perform inner without have actual data fram view
    merged_df = pd.merge(df, dfUpdateData, left_on=view_name_id, right_on='id', how='inner')
    merged_df = merged_df.drop(columns=['id'])

    listAllAction=dfUpdateData['id'].tolist()
    print(f"List {listAllAction} all action")
    
    listSeleted = merged_df[view_name_id].tolist()
    print(f"List  {x_view}   {listSeleted} from {x_view} exluding deleted action")
    
    allActionSet = set(listAllAction)
    anotherSet = set(listSeleted)
    
    listDeleted = list(allActionSet.symmetric_difference(anotherSet))
    print(f"List deleted {listDeleted}")
    
    # Test List  select by view + List deeleted = List All Action

    if len(listDeleted)>0:
        print("There are some deleted rows")
        dfDeleted=pd.DataFrame(data=listDeleted,columns=[view_name_id])
        dfDeleted['action']='deleted'
        print(dfDeleted)
        merged_df=pd.concat([merged_df,dfDeleted],axis=0)

    else:
        print("No row deleted")

    return merged_df    




# In[106]:


if isFirstLoad==False:
 df=add_acutal_action_to_df_at_next(df,dfModelLog,view_name,view_name_id)

print(df)


# In[ ]:





# # Last Step :Check duplicate ID & reset index

# In[107]:


hasDplicateIDs = df[view_name_id].duplicated().any()
if  hasDplicateIDs:
 raise Exception("There are some duplicate id on dfUpdateData")
else:
 print(f"There is no duplicate {view_name_id} ID")  


# merged_df['imported_at']=dt_imported
df=df.reset_index(drop=True  )
print(df.info())
print(df)


# In[108]:


df


# # Insert data to BQ data frame

# In[109]:


if get_bq_table():
    try:
        insertDataFrameToBQ(df)
    except Exception as ex:
        raise ex


# # Run StoreProcedure To Merge Temp&Main and Truncate Transaction 

# In[110]:


print("# Run StoreProcedure To Merge Temp&Main and Truncate Transaction.")
# https://cloud.google.com/bigquery/docs/transactions
sp_id_to_invoke=f""" CALL `{projectId}.{main_dataset_id}.{sp_name}`() """
print(sp_id_to_invoke)

sp_job = client.query(sp_id_to_invoke)


# In[111]:


updater["metadata"][view_name].value=dt_imported.strftime("%Y-%m-%d %H:%M:%S")
updater.update_file() 


# In[112]:


print(datetime.now(timezone.utc) )


# In[ ]:





# 

# In[ ]:




