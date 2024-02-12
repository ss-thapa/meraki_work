import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import sys

pd.set_option('display.max_columns', None)



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')

col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])

final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

daily_data = final_df.resample('D', on='ds').sum().reset_index()

daily_data = daily_data[daily_data['ds'].dt.strftime('%Y-%m-%d') != '2024-02-03']

saturdays = pd.date_range(start=final_df['ds'].min(), end=final_df['ds'].max(), freq='W-SAT')

holidays_df = pd.DataFrame({'holiday': 'Saturday', 'ds': saturdays})

daily_data = daily_data.merge(holidays_df, on='ds', how='left')

daily_data['holiday'] = daily_data['holiday'].fillna(0)



mod = Prophet(weekly_seasonality=20,daily_seasonality=3)

mod.add_regressor('holiday')

model = mod.fit(daily_data)

# Make future predictions
future_dates = pd.date_range(start=daily_data['ds'].max()+ pd.Timedelta(days=1), periods=30, freq='D')  # Generate future dates
future = pd.DataFrame({'ds': future_dates})

# Indicate Saturdays as holidays for future dates
future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)

# Make predictions
forecast = model.predict(future)

# print(forecast[['ds', 'yhat']])


model_size = sys.getsizeof(model)

print(f"Estimated size of the Prophet model object: {model_size} bytes")
