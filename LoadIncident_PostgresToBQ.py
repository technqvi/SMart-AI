#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import psycopg2.extras as extras
import pandas as pd
import json
from datetime import datetime,timezone

from dotenv import dotenv_values

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest
from google.oauth2 import service_account

from dotenv import dotenv_values
# https://cloud.google.com/bigquery/docs/quickstarts/quickstart-client-libraries
# https://codelabs.developers.google.com/codelabs/cloud-bigquery-python#0

# https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-console
# https://cloud.google.com/python/docs/setup
# https://cloud.google.com/apis/docs/getting-started#enabling_apis

# https://cloud.google.com/bigquery/docs/reference/libraries
# https://cloud.google.com/python/docs/reference/bigquery/latest
# https://gcloud.readthedocs.io/en/latest/bigquery-client.html


# In[2]:


# credentials = service_account.Credentials.from_service_account_file(r'C:\Windows\xxxxxx.json')
# print(credentials.project_id)


# In[4]:


start_date_query='2020-01-01'

projectId='smart-data-ml'  # 
dataset_id='SMartDW'
table_id = f"{projectId}.{dataset_id}.incident"

env_path=r'E:\SMartAppScripts\.env'
config = dotenv_values(dotenv_path=env_path)



# In[5]:


# client = bigquery.Client(credentials= credentials,project=projectId)
client = bigquery.Client(project=projectId)


# In[9]:


# D:\PythonDev\MyQuantFinProject\SMart-AI\data\Incident_PostgresToBQ_Schema.txt
try:
    table=client.get_table(table_id)  # Make an API request.
    print("Table {} already exists.".format(table_id))
    print(table.schema)
except NotFound:
    raise Exception("Table {} is not found, please manually create table named incident including partition and clustering".format(table_id))


# In[10]:


dt_imported=datetime.now(timezone.utc) # utc
#dt_imported=datetime.now() # utc
str_imported=dt_imported.strftime('%Y-%m-%d %H:%M:%S')
print(f"Imported DateTime: {str_imported}" )


# In[11]:


sql_lastImport=f"SELECT max(imported_at) as last_imported from `{table_id}` where open_datetime>='{start_date_query}' "
job_lastImported=client.query(sql_lastImport)
str_lastImported=None
for row in job_lastImported:    
    if row.last_imported is not None: 
        str_lastImported=row.last_imported.strftime('%Y-%m-%d %H:%M:%S')
print(f"Last Imported DateTime: {str_lastImported}" )

if str_lastImported is not None:
  start_date_query=str_lastImported

print(f"Start Import on update_at of last imported date : {start_date_query}" )


# In[12]:


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


# In[13]:


# check schema 
#,timezone('UTC', incident.updated_at) as updated_at_utc
# and incident.updated_at>=%(start_date_param)s
sql_incident=f"""

select
incident.id as id, incident.incident_no as incident_no,

severity.id as  severity_id,
severity.severity_name as  severity_name,

service_level.sla_name as sla,

product_type.productype_name as product_type,brand.brand_name as brand,

xtype.incident_type_name as incident_type,
status.incident_status_name as status,
service.service_type_name service_type,

model.model_name as model,CASE WHEN failure_type IS NULL THEN  0 ELSE 1 END AS is_failure_type,


(select count(*) from  app_incident_detail  as detail where  detail.incident_master_id=incident.id ) as count_detail


,TO_CHAR(incident.incident_datetime  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as open_datetime
,TO_CHAR(incident.incident_close_datetime  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as close_datetime

,TO_CHAR(incident.incident_problem_start  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as response_datetime
,TO_CHAR(incident.incident_problem_end  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as resolved_datetime

,company.company_name as company
,     (SELECT emp.employee_name
        FROM app_employee emp
        WHERE emp.id = incident.incident_owner_id) AS case_owner
,TO_CHAR(incident.updated_at,'YYYY-MM-DD HH24:MI:SS') as updated_at 

from app_incident as incident
inner join app_incident_type as  xtype on incident.incident_type_id = xtype.id
inner join  app_incident_status as status on incident.incident_status_id = status.id
inner join  app_incident_severity as severity on  incident.incident_severity_id = severity.id
inner join  app_service_type as service on incident.service_type_id= service.id

inner join app_inventory as inventory on incident.inventory_id = inventory.id

inner join app_brand as brand on inventory.brand_id = brand.id
inner join app_model as model on inventory.model_id = model.id
inner join app_product_type as product_type on inventory.product_type_id = product_type.id
inner join app_sla as service_level on inventory.customer_sla_id = service_level.id

inner join app_project as project on inventory.project_id = project.id
inner join app_company as company on project.company_id = company.id

where incident.incident_status_id =4
and timezone('UTC', incident.updated_at)>='{start_date_query}'

order by incident.updated_at 

"""

# and timezone('UTC', incident.updated_at)>=%(start_date_param)s
print(sql_incident)


# In[15]:


print("Create all issues dataframe")
#and timezone('UTC', incident.updated_at)>=%(start_date_param)s
# dict_params={"start_date_param":start_date_query}
# df_all=list_data(sql_incident,dict_params,get_postgres_conn())


df_all=list_data(sql_incident,None,get_postgres_conn())

if df_all.empty==True:
    print("no transsaction update")
    exit()
else:
    print("import data to bigquery")
print(df_all.info())
df_all.tail()


# In[16]:


# convert object to datetime
dateTimeCols=['open_datetime','response_datetime','resolved_datetime','close_datetime']
for col in dateTimeCols:
 df_all[col]=pd.to_datetime(df_all[col], format='%Y-%m-%d %H:%M',errors= 'coerce')

dateTimeCols2=['updated_at']
for col in dateTimeCols2:
 df_all[col]=pd.to_datetime(df_all[col], format='%Y-%m-%d %H:%M:%S',errors= 'coerce')

# UTC Import
df_all['imported_at']=dt_imported


# In[17]:


df_all.dropna(inplace=True)

#df_all=df_all.head(10)
print(df_all.info())
df_all.head(10)


# In[18]:


def insertDataFrameToBQ(df_trasns):
    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        job = client.load_table_from_dataframe(
            df_trasns, table_id, job_config=job_config
        )
        job.result()  # Wait for the job to complete.
        print("Total ", len(df_trasns), "Imported closed incident to bigquery successfully")

    except BadRequest as e:
        print("Bigquery Error\n")
        print(e) 

try:
    insertDataFrameToBQ(df_all)
except Exception as ex:
    raise ex




# In[19]:


importDate=dt_imported.strftime("%Y-%m-%d")

def load_data_bq(sql:str):
 query_result=client.query(sql)
 dfx=query_result.to_dataframe()
 return dfx
sql_query=f""" SELECT * FROM `{table_id}` WHERE imported_at >= '{importDate}' LIMIT 10 """
df=load_data_bq(sql_query)
df.head()


# In[ ]:





# In[ ]:




