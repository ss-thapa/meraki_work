import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import plotly.express as px




pd.set_option('display.max_columns', None)



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')


col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])

final_df = final_df[final_df['grand_total'] < 20000]
final_df = final_df[final_df['grand_total'] != 0]


final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

final_df = final_df[final_df['ds'].dt.strftime('%Y-%m-%d') != '2024-02-03']

saturdays = pd.date_range(start=final_df['ds'].min(), end=final_df['ds'].max(), freq='W-SAT')

holidays_df = pd.DataFrame({'holiday': 'Saturday', 'ds': saturdays})

final_df =final_df.merge(holidays_df, on='ds', how='left')

final_df['holiday'] =final_df['holiday'].fillna(0)




mod = Prophet(weekly_seasonality=20,daily_seasonality=3)

mod.add_regressor('holiday')

model = mod.fit(final_df)

# Generate future dates on hourly basis
future_dates = pd.date_range(start=final_df['ds'].max()+ pd.Timedelta(days=1), periods=24*8, freq='H')  
future = pd.DataFrame({'ds': future_dates})

future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)

forecast = model.predict(future)



forecast_2024_02_04 = forecast[(forecast['ds'] >= '2024-02-04 00:00:00') & (forecast['ds'] < '2024-02-05 00:00:00')]
print(forecast_2024_02_04)
yhat_predicted_for_day = forecast_2024_02_04['yhat'].sum()




# print(forecast)

# future = model.make_future_dataframe(periods=24*7, freq='h', include_history=False)
# forecast = model.predict(future)




# forecast_2024_02_04 = forecast[(forecast['ds'] >= '2024-02-05 00:00:00') & (forecast['ds'] < '2024-02-06 00:00:00')]
# yhat_predicted_for_day = forecast_2024_02_04['yhat'].sum()

# print(forecast_2024_02_04)



