import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.preprocessing import StandardScaler

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





#### crossvalidation and error metrics


train_data = daily_data[daily_data['ds'] < '2024-01-01']

test_data = daily_data[daily_data['ds'] >= '2024-01-01']

# sns.lineplot(x=train_data['ds'],y=train_data['y'])
# plt.show()


# print(train_data)

# sns.boxplot(train_data['y'])
# plt.show()


mod = Prophet()

mod.add_regressor('holiday')

model = mod.fit(train_data)

days_to_forecast = len(test_data)

# Make future predictions
future_dates = pd.date_range(start=train_data['ds'].max() + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')  # Generate future dates
future = pd.DataFrame({'ds': future_dates})

# Indicate Saturdays as holidays for future dates
future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)

# Make predictions
forecast = model.predict(future)

forecast_values = forecast[-len(test_data):]['yhat']



## using different matrics



mape = mean_absolute_percentage_error(test_data['y'], forecast_values)

print(mape)




# trace_pred = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted')
# trace_actual = go.Scatter(x=test_data['ds'], y=test_data['y'], mode='lines', name='Actual')

# # Create layout
# layout = go.Layout(title='Predicted vs Actual',
#                    xaxis=dict(title='Date'),
#                    yaxis=dict(title='Value'))

# # Create figure and add traces
# fig = go.Figure(data=[trace_pred, trace_actual], layout=layout)

# # Plot the figure
# py.plot(fig, filename='predicted_vs_actual.html')



