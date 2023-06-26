#!/usr/bin/env python
# coding: utf-8

# In[4]:


from google.cloud import bigquery
import pandas as pd
from datetime import date,datetime,timedelta
from google.cloud import bigquery
from google.oauth2 import service_account


# In[5]:


isLoadingAll=False

# we synch at 02:30(Bangkok) it is  7:30 (UTC is still Yesterday)
predict_datetime = date.today() - timedelta(days = 1) 
#predict_datetime = date.today()
print(f"Get prediction as of {predict_datetime}")


# In[6]:


projectId='smart-data-ml'
table_id = f"{projectId}.SMartML.new_result_prediction_incident"

#credentials = service_account.Credentials.from_service_account_file(r'C:\Windows\smart-data-ml-91b6f6204773.json')
#client = bigquery.Client(credentials= credentials,project=projectId)
client = bigquery.Client(project=projectId)


# In[7]:


def load_data_bq(sql:str):

 query_result=client.query(sql)
 df_all=query_result.to_dataframe()
 return df_all

if isLoadingAll:
    sql=f""" select id,predict_severity,prediction_datetime from {table_id} """
else:
    sql=f"""
    select id,predict_severity,prediction_datetime from {table_id} 
     where Date(prediction_datetime)='{predict_datetime}' """
    
print(sql)

df=load_data_bq(sql)
if len(df)>0:
    df.columns=['incident_id','severity_label','prediction_at']
else:
    quit()


# In[8]:


# load json file
map_sevirity_to_class={'Cosmetic':0,'Minor': 1, "Major": 2, "Critical": 3}
revert_class_to_severity = {v: k for k, v in map_sevirity_to_class.items()}

print(f"Map severity Name to LabelCode: {str(revert_class_to_severity)}")

df['severity_name']=df['severity_label'].map(revert_class_to_severity)

update_at=datetime.now()
df['imported_at']=update_at


# In[9]:


print(df.info())
print(df)


# In[10]:


import psycopg2
import psycopg2.extras as extras

def get_postgres_conn():
 try:
  conn = psycopg2.connect(
         database='SMartDB', user='postgres',
      password='P@ssw0rd', host='localhost', 
     )
  return conn

 except Exception as error:
  print(error)      
  raise error

def add_data_values(df, table,conn):

    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    #print(query)
    #return query,tuples
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        raise error
        return 0
    
    return 1
    cursor.close()
    
result=add_data_values(df,'app_prediction_ml_severity_incident',get_postgres_conn())

if  result==1:
    print(f"{len(df.index)} items have been imported to database successfully.")
    print("importing data succeeded")


# In[ ]:




