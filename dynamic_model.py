import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import sys
from prophet.plot import plot,add_changepoints_to_plot

pd.set_option('display.max_columns', None)



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')


col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])

final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

daily_data = final_df.resample('D', on='ds').sum().reset_index()

daily_data = daily_data[daily_data['ds'].dt.strftime('%Y-%m-%d') != '2024-02-03']

daily_data['holiday'] = (daily_data['ds'].dt.dayofweek == 5).astype(int)




mod = Prophet(changepoint_prior_scale=0.1)

mod.add_regressor('holiday')

model = mod.fit(daily_data)

days_to_forecast = 30

# Make future predictions
future_dates = pd.date_range(start=daily_data['ds'].max()+ pd.Timedelta(days=1), periods=days_to_forecast, freq='D')  # Generate future dates
future = pd.DataFrame({'ds': future_dates})

# Indicate Saturdays as holidays for future dates
future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)

# Make predictions
forecast = model.predict(future)


forecast[['ds', 'yhat']]




change_points = model.changepoints

# Plot the data along with the change points
plt.figure(figsize=(10, 6))
plt.plot(daily_data['ds'], daily_data['y'], label='Actual Data')
plt.vlines(change_points, ymin=daily_data['y'].min(), ymax=daily_data['y'].max(), colors='r', linestyles='dashed', label='Change Points')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Change Points Detected by Prophet')
plt.legend()
plt.show()



