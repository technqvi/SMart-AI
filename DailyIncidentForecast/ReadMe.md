# About

<img width="689" alt="image" src="https://github.com/technqvi/SMart-AI/assets/38780060/45bdd3db-bbd9-418e-8b1c-036250f0b020">

Reference: [Building LSTM Time Series model to predict future stock price movement](https://github.com/technqvi/TimeSeriesML-FinMarket/tree/main/forecast-asset%20-price-movement-LSTM-TimeSeries)


* [LoadDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/LoadDailyIncident.ipynb ) :Import the number of daily incident from SMartApp Database  to BigQuery on daily basis
* [BuildModelNoDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/BuildModelNoDailyIncident.ipynb) : Tune and Build LSTM  Time-Serice Model in order to  retrieve the number of daily incident over the past x days to make prediction for the next y days by LSTM Time-Serice Model.
* [ForecastNoDailyIncident.ipynb](https://github.com/technqvi/SMart-AI/blob/main/DailyIncidentForecast/ForecastNoDailyIncident.ipynb) : Load model to marke Prediction  on the last the number of incident and store prediction result on BigQuery
* To visulaize prediction the number of incident  a head of time , you can apply PowerBI as dashboard tools.