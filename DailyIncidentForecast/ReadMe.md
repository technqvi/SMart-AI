# About
This project involves in building time series model using LSTM on Tenforflow framework to forecast the number of daily incident.
For more detail about how to develop the project, please refer to  [Building LSTM Time Series model to predict future stock price movement](https://github.com/technqvi/TimeSeriesML-FinMarket/tree/main/forecast-asset%20-price-movement-LSTM-TimeSeries)

<img width="689" alt="image" src="https://github.com/technqvi/SMart-AI/assets/38780060/45bdd3db-bbd9-418e-8b1c-036250f0b020">




* [LoadDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/LoadDailyIncident.ipynb ) :Import the number of daily incident from SMartApp Database  to BigQuery on daily basis
* [BuildModelNoDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/BuildModelNoDailyIncident.ipynb) : Tune and Build LSTM  Time-Serice Model in order to  retrieve the number of daily incident over the past x days to make prediction for the next y days by LSTM Time-Serice Model.
* [ForecastNoDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/ForecastNoDailyIncident.ipynb) : Load model to marke Prediction  on the last the number of incident and store prediction result on BigQuery
* [CollectForecastPerformance.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/CollectForecastPerformance.ipynb) : Collect model performance data on weekly basis by retrieving the last 2 week forcasting compared to the number of actual incidens  and calculate MAE metric, finally the data will be stored into BigQuery in order for PowerBI to pull data to visualize the performance collection data.
* To visulaize prediction the number of incident , actual value compared to predicted value and MAE value of each weekly data collection , you can apply PowerBI as dashboard tools.