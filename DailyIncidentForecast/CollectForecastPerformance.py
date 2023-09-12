#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import os
from datetime import datetime,date,timedelta,timezone
import calendar
import json


from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import BadRequest



# In[81]:


# uncomment and indent
# collectionDate='2023-09-10 01:00' # comment
mode=2
model_id="Incident_60To5_E150S15B32-M0122-0723"
n_day_ago_to_lookback=7

def collect_prediction_result(collectionDate=None):



    # # Init parameter

    # In[82]:


    if mode==1: # Migrate to backfill data and Test 
        logDate=collectionDate
        log_date=datetime.strptime(logDate,'%Y-%m-%d %H:%M')
        log_timestamp=datetime.strptime(logDate,'%Y-%m-%d %H:%M')
    else: # On weekly basis
        log_timestamp=datetime.now()
        log_date=datetime.strptime(log_timestamp.strftime('%Y-%m-%d'),'%Y-%m-%d')

    week_day=log_date.weekday()
    day_name=calendar.day_name[log_date.weekday()]

    print(f"Date to collect data on {log_date.strftime('%Y-%m-%d')} {day_name}(Idx:{week_day}) at {log_timestamp}")
    if  week_day!=6:
     raise Exception("Sunday is allowed  as Collection Date for forcasting result.")   


    genTableSchema=False
    metric_name='mae'

    date_col='date'



    # # Create Start to End Date By Getting Last Date of Week

    # In[83]:


    # get  prev prediction  from  get end prediction to beginneg or predicton of week 
    endX=log_date+timedelta(days=-n_day_ago_to_lookback)# Friday 2 week ago
    startX=endX+timedelta(days=-n_day_ago_to_lookback+1)# Monday 2 week ago
    # print(f"Collection data the last  {(endX-startX).days+1} days :From {startX.strftime('%A %d-%m-%Y')} To {endX.strftime('%A %d-%m-%Y')}")

    endX=endX.strftime('%Y-%m-%d')
    startX=startX.strftime('%Y-%m-%d')

    print(f"Convert start and end data {startX} - {endX} to string")


    # # BigQuery Setting & Configuration Variable

    # In[84]:


    projectId='smart-data-ml'
    dataset_id='SMartDW'

    table_data_id=f"{projectId}.{dataset_id}.daily_incident"
    table_id = f"{projectId}.{dataset_id}.prediction_daily_incident"
    table_perf_id= f"{projectId}.{dataset_id}.model_perfromance_daily_incident"

    print(table_id)
    print(table_data_id)
    print(table_perf_id)

    # client = bigquery.Client(project=projectId )
    credentials = service_account.Credentials.from_service_account_file(r'C:\Windows\smart-data-ml-91b6f6204773.json')
    client = bigquery.Client(credentials=credentials, project=projectId)

    def load_data_bq(sql:str):
        query_result=client.query(sql)
        df=query_result.to_dataframe()
        return df


    # # Check where the given date collected data or not?

    # In[85]:


    sqlCheck=f"""
    select collection_timestamp from `{table_perf_id}`
    where date(collection_timestamp)='{log_date.strftime('%Y-%m-%d')}' and model_id='{model_id}'
    """

    print(sqlCheck)
    dfCheckDate=load_data_bq(sqlCheck)
    if  dfCheckDate.empty==False:
        print(f"Collection data on {log_date} for {model_id} found, no any action")
        # uncomment
        return f"Collection data on {log_date} for {model_id} found, no any action"
    else:
        print(f"We are ready to Collect data on {log_date}")


    # # Retrive forecasting result data to Dictionary

    # In[86]:


    def get_forecasting_result_data(request):

        if   request is not None:  
            start_date=request["start_date"]
            end_date=request["end_date"]
            model_id=request["model_id"]
        else:
            raise Exception("No request parameters such as start_date,end_date")

        
        print("1.How far in advance does model want to  make prediction")
        sqlOutput=f"""
        select t.pred_timestamp,t.prediction_date,t_pred.date,t_pred.count_incident
        from  `{table_id}` t cross join unnest(t.prediction_result) t_pred
        where (t.prediction_date>='{start_date}' and  t.prediction_date<='{end_date}')
        and t.model_id='{model_id}' and t_pred.type='prediction'
        order by  t.pred_timestamp,t.prediction_date,t_pred.date
        """
        print(sqlOutput)
        dfOutput=load_data_bq(sqlOutput)
        dfOutput[date_col]=pd.to_datetime(dfOutput[date_col],format='%Y-%m-%d')
        dfOutput.set_index(date_col,inplace=True)

        output_sequence_length=len(dfOutput)
        print(f"output_sequence_length={output_sequence_length}")
        

        print(dfOutput.info())
        print(dfOutput)
        print("================================================================================================")

        
        #get actual data since the fist day of input and the last day of output(if covered)
        startFinData=dfOutput.index.min().strftime('%Y-%m-%d')
        endFindData=dfOutput.index.max().strftime('%Y-%m-%d')
        print(f"2.Get Real Data  to compare to prediction from {startFinData} to {endFindData}")

        sqlData=f"""
        select {date_col},count_incident, datetime_imported, from `{table_data_id}` 
        where ({date_col}>='{startFinData}' and {date_col}<='{endFindData}')
        order by datetime_imported,{date_col}
        """
        
        print(sqlData)

        dfRealData=load_data_bq(sqlData)
        dfRealData=dfRealData.drop_duplicates(subset=[date_col],keep='last',)
        dfRealData[date_col]=pd.to_datetime(dfRealData[date_col],format='%Y-%m-%d')
        dfRealData.set_index(date_col,inplace=True)
        
        print(dfRealData.info())
        print(dfRealData)
        print("================================================================================================")

        return {'actual_no_incident':dfRealData,'predicted_no_incident':dfOutput }


    print(f"================Get data from {startX}====to==={endX}================")
    request={'start_date':startX,'end_date':endX,'model_id':model_id}
    data=get_forecasting_result_data(request)
    print(f"=======================================================================")


    # # Create Predictive and Actual Value dataframe

    # In[87]:


    print("List all trading day in the week")
    myTradingDataList=data['predicted_no_incident']['prediction_date'].unique()
    print(myTradingDataList)


    # In[88]:


    print(f"========================dfX :Actual Price========================")
    dfX=data['actual_no_incident'][['count_incident']]
    dfX.columns=[f'actual_value']
    print(dfX.info())
    print(dfX)


    # In[89]:


    dfAllForecastResult=pd.DataFrame(columns=['date','pred_value','actual_value','prediction_date'])
    # actually , we can jon without spilting data by prediction_dtate
    for date in  myTradingDataList: # trading day on giver week
        print(f"=========================dfPred:Predicted Price at {date}=========================")
        dfPred=data['predicted_no_incident'].query("prediction_date==@date")[['count_incident']]
        dfPred.columns=[f'pred_value']
        # print(dfPred.info())

        print("=====================dfCompare:Join Actual price to Predicted Price=================")
        dfCompare=pd.merge(left=dfPred,right=dfX,how='inner',right_index=True,left_index=True)
        dfCompare.reset_index(inplace=True)   
        dfCompare['prediction_date']=date.strftime('%Y-%m-%d')      
        print(dfCompare) 
        print(dfCompare.info())

        if len(dfCompare)>0 : # it will be join if there is at least one record to show actual vs pred
            dfAllForecastResult= pd.concat([dfAllForecastResult,dfCompare],ignore_index=True)
            print(f"=========================Appended Data Joined=========================")
        else:
            print("No Appendind Data due to no at least one record to show actual vs pred")


    # In[90]:


    print("========================dfAllForecastResult: All Predicton Result========================")
    dfAllForecastResult[date_col]=dfAllForecastResult[date_col].dt.strftime('%Y-%m-%d')
    print(dfAllForecastResult.info())
    print(dfAllForecastResult)


    # # Calculate Metric

    # ## Get sum distance between pred and actul value from prev rows

    # In[91]:


    sqlMetric=f"""
    with pred_actual_by_model as  
    (
    SELECT  detail.actual_value,detail.pred_value
    from `{table_perf_id}`  t
    cross join unnest(t.pred_actual_data) as detail
    where t.model_id='{model_id}' and t.collection_timestamp<'{log_timestamp}'
    )
    select COALESCE( sum(abs(x.actual_value-x.pred_value)),0) as pred_diff_actual,count(*) as no_row  from pred_actual_by_model  x


    """

    if genTableSchema==False:
        print(sqlMetric)

        dfMetric=load_data_bq(sqlMetric)
        prevSum=dfMetric.iloc[0,0]
        prevCount=dfMetric.iloc[0,1]

    else:  # it is used if there are something changed in table schema
    # for generating table schema
        prevSum=0
        prevCount=0

    print(f"Prev Sum={prevSum} and Count={prevCount}")


    # ## Cal sum distance between pred and actul value from last rows

    # In[92]:


    dfAllForecastResult['pred_diff_actual']=dfAllForecastResult.apply(lambda x : abs(x['pred_value']-x['actual_value']),axis=1)
    recentSum=dfAllForecastResult['pred_diff_actual'].sum()
    recentCount=len(dfAllForecastResult)

    dfAllForecastResult=dfAllForecastResult.drop(columns=['pred_diff_actual'])
    print(f"Recent Sum={recentSum} and Count={recentCount}")

    #https://en.wikipedia.org/wiki/Mean_absolute_error
    metric_value= round((prevSum+recentSum)/(prevCount+recentCount),2)
    print(f"{metric_name} = {metric_value}")


    # # Create Collection Performance Info Dataframe and Store 
    # 

    # In[93]:


    masterDF=pd.DataFrame(data=[ [log_date,model_id,metric_name,metric_value,log_timestamp] ],
                    columns=["collection_date","model_id","metric_name","metric_value","collection_timestamp"])

    masterDF["collection_date"]=masterDF["collection_date"].dt.strftime('%Y-%m-%d') # for json format
    masterDF["collection_timestamp"]=masterDF["collection_timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S') # for json format
    print(masterDF.info())
    masterDF


    # # Create Dataframe to  Json Data 

    # In[94]:


    master_perf = json.loads(masterDF.to_json(orient = 'records')) # 1 main dataframe has 1 records
    for  master in master_perf:
        detail= json.loads(dfAllForecastResult.to_json(orient = 'records'))
        master["pred_actual_data"]=detail

        
    # with open("no_incident_forecast_performance.json", "w") as outfile:
    #     json.dump( master_perf, outfile)


    # In[ ]:





    # # Ingest Data to BigQuery

    # ## Try to ingest data to get correct schema and copy the schema to create table including partion/cluster manually

    # In[95]:


    try:
        table=client.get_table(table_perf_id)
        print("Table {} already exists.".format(table_id))
        
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        # Try to ingest data to get correct schema and copy the schema to create table including partiion/cluster manually
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND 
        job = client.load_table_from_json(master_perf,table_perf_id, job_config = job_config)
        if job.errors is not None:
            print(job.error_result)
            print(job.errors)
            # uncomment
            return "Error to load data to BigQuery"
        else:
            print(f"Import to bigquery successfully  {len(master_perf)} records")
            # print(table.schema)
    except Exception as ex :
        print(str(ex))
        

        
    #job_config.schema
    # truncate table `smart-data-ml.SMartDW.model_perfromance_daily_incident`  


    # In[165]:


    # uncomment
    return 'completely'


# In[ ]:





# In[97]:


# mode2 # weekly
collect_prediction_result()

# mode1 # backfill
# start_backfill='2023-07-16 01:00' # comment
# end_backfill='2023-07-30 01:00'

# start_backfill='2023-08-06 01:00'
# end_backfill='2023-09-10 01:00'

# period_index=pd.date_range(start=start_backfill,end=end_backfill, freq="W-SUN")
# listLogDate=[ d.strftime('%Y-%m-%d %H:%M')   for  d in  period_index   ]
# print(listLogDate)
# for d in listLogDate:
#     print(f"###################################Start Collecting Data on {d}###################################")
#     collect_prediction_result(d)
#     print("###################################End###################################")
    



# In[ ]:




