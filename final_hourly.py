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

# final_df = final_df[final_df['grand_total'] < 20000]
# final_df = final_df[final_df['grand_total'] != 0]


final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

final_df = final_df[final_df['ds'].dt.strftime('%Y-%m-%d') != '2024-02-03']

final_df['holiday'] = (final_df['ds'].dt.dayofweek == 5).astype(int)

final_df = final_df[(final_df['ds'].dt.hour >= 9) & (final_df['ds'].dt.hour <= 20)]



# print(final_df[final_df['ds'].dt.strftime('%Y-%m-%d') == '2024-01-04'])


mod = Prophet()

mod.add_regressor('holiday')

model = mod.fit(final_df)

# # Generate future dates on hourly basis

future = model.make_future_dataframe(periods=200, freq='H')
future['ds'] = pd.to_datetime(future['ds'])
future = future[(future['ds'] >= pd.Timestamp('2024-02-05 09:00:00')) & (future['ds'] <= pd.Timestamp('2024-02-05 20:00:00'))]


future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)




forecast = model.predict(future)

print(forecast['yhat'].sum())

# model.plot(forecast)
# plt.show()



