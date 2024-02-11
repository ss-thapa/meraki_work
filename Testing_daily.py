import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



pd.set_option('display.max_columns', None)



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')
df_sales_detail = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_detail.csv')
# df_purchas_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/purchase_main.csv')
# df_purchase_detail = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/purchase_detail.csv')
# df_item = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/items.csv')



col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])


final_df =final_df[final_df['grand_total'] < 20000]

final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

daily_data = final_df.resample('D', on='ds').sum().reset_index()


print(daily_data.tail())

# mod = Prophet()

# model = mod.fit(daily_data)

# future = model.make_future_dataframe(periods=10, freq='D', include_history=False)
# forecast = model.predict(future)

# print(forecast)






