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



# plt.figure(figsize=(15,6))
# plt.plot(final_df['created_date_ad'],final_df['grand_total'])
# plt.xticks(rotation='vertical')
# plt.show()

final_df =final_df.rename(columns={'created_date_ad':'ds','grand_total':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)

final_df = final_df.set_index('ds')




split_date = '2024-01-01'
final_df_train = final_df.loc[final_df.index <= split_date].copy()
final_df_test = final_df.loc[final_df.index > split_date].copy()


# # Plot train and test so you can see where we have split
# final_df_test \
#     .rename(columns={'y': 'TEST SET'}) \
#     .join(fina_df_train.rename(columns={'y': 'TRAINING SET'}),
#           how='outer') \
#     .plot(figsize=(10, 5), title='PJM East', style='.', ms=1)
# plt.show()


# Format data for prophet model using ds and y
final_df_train = final_df_train.reset_index() \
    .rename(columns={'ds':'ds',
                     'y':'y'})

final_df_test = final_df_test.reset_index() \
    .rename(columns={'ds':'ds',
                     'y':'y'})


# ### training the model in train dataset

mod = Prophet()

model = mod.fit(final_df_train)


# ## Predict on test set with model
df_test_prophet = final_df_test.reset_index().rename(columns={'Datetime':'ds','PJME_MW':'y'})

final_prediction_df = model.predict(df_test_prophet)



### make future prediction 


future = model.make_future_dataframe(periods=24, freq='H', include_history=False)
forecast = model.predict(future)

forecast_jan_1_2024 = forecast[forecast['ds'].dt.date == pd.Timestamp('2024-01-01')]

print(forecast_jan_1_2024)
# daily_forecast = forecast.groupby(forecast['ds'].dt.date)['yhat'].sum().reset_index()

# print(daily_forecast)






# ##plotting 
# # fig, ax = plt.subplots(figsize=(10, 5))
# # fig = model.plot(final_prediction_df, ax=ax)
# # ax.set_title('Prophet Forecast')
# # plt.show()

# ### plotting daily weekly trend 
# # fig = model.plot_components(final_prediction_df)
# # plt.show()


# ### ploting predicted and actual values

# f, ax = plt.subplots(figsize=(15, 5))
# ax.scatter(final_df_test['ds'], final_df_test['y'], color='r')
# fig = model.plot(final_prediction_df, ax=ax)
# plt.show()




### plotting within date actual and forcasted of 1 week 

# lower_bound = pd.to_datetime('2024-01-02')
# upper_bound = pd.to_datetime('2024-01-10')

# # Plot the forecast with the actuals
# f, ax = plt.subplots(figsize=(15, 5))
# ax.scatter(final_df_test['ds'], final_df_test['y'], color='r')
# fig = model.plot(final_prediction_df, ax=ax,include_legend=True)
# ax.set_xbound(lower_bound, upper_bound)
# ax.set_title('First Week of January Forecast vs Actuals')
# plt.show()




### plotting within date actual and forcasted of 1 month

# lower_bound = pd.to_datetime('2024-01-02')
# upper_bound = pd.to_datetime('2024-02-02')

# # Plot the forecast with the actuals
# f, ax = plt.subplots(figsize=(15, 5))
# ax.scatter(final_df_test['ds'], final_df_test['y'], color='r')
# fig = model.plot(final_prediction_df, ax=ax,include_legend=True)
# ax.set_xbound(lower_bound, upper_bound)
# ax.set_title('First Week of January Forecast vs Actuals')
# plt.show()



### accuracy checking with mean squared value

# print(np.sqrt(mean_squared_error(y_true=final_df_test['y'],y_pred=final_prediction_df['yhat'])))


### accuracy checking with mean_absolute_percentage_error

# print(np.sqrt(mean_absolute_percentage_error(y_true=final_df_test['y'],y_pred=final_prediction_df['yhat'])))






