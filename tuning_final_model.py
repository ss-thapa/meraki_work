import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objs as go
import plotly.offline as py
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics



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


train_data = daily_data[daily_data['ds'] < '2024-01-15']

test_data = daily_data[daily_data['ds'] >= '2024-01-15']



mod = Prophet(n_changepoints=35,changepoint_prior_scale=0.1,daily_seasonality=False,yearly_seasonality=False,weekly_seasonality=False,seasonality_prior_scale=0)

mod.add_regressor('holiday')

model = mod.fit(train_data)

days_to_forecast = len(test_data)

# Make future predictions
future_dates = pd.date_range(start=train_data['ds'].max() + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')
future = pd.DataFrame({'ds': future_dates})

# Indicate Saturdays as holidays for future dates
future['holiday'] = (future['ds'].dt.dayofweek == 5).astype(int)

# Make predictions
forecast = model.predict(future)

forecast_values = forecast[-len(test_data):]['yhat']


### mape matrics

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







## cross validation 


# # Initialize Prophet model
# model = Prophet(changepoint_prior_scale=0.2505,seasonality_prior_scale=2.5075)
# model.add_regressor('holiday')
# model.fit(daily_data)


# # Perform cross-validation
# df_cv = cross_validation(model, horizon='30 days', initial='210 days', period='30 days')

# # Compute performance metrics
# df_metrics = performance_metrics(df_cv)
# print(df_metrics.head())











# changepoint_prior_scale_range = np.linspace(0.001, 0.5, num=2).tolist()

# seasonality_prior_scale_range = np.linspace(0.01, 10, num=2).tolist()

# holidays_prior_scale_range = np.linspace(0.01, 10, num=2).tolist()


# category_df = daily_data.copy()
# category_df.columns = ["ds", "y"]
# category_df[["y"]] = category_df[["y"]].apply(pd.to_numeric)
# category_df["ds"] = pd.to_datetime(category_df["ds"])

# # Start timer
# start_time = time.time()

# # Initialize dictionary to store results
# dicts = {}

# # Generate all combinations of hyperparameters
# param_grid = {
#     "changepoint_prior_scale": changepoint_prior_scale_range,
#     "seasonality_prior_scale": seasonality_prior_scale_range
# }
# all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]



# mapes = []
# for params in all_params:
#     m = Prophet(**params).fit(category_df)  # Fit model with given params
#     df_cv = cross_validation(m, initial="180 days", period="30 days", horizon="30 days")  # Perform cross-validation
#     df_p = df_cv.groupby('cutoff').apply(performance_metrics, rolling_window=1)
#     mapes.append(df_p["mape"].values.mean())

# # Find the best parameters
# tuning_results = pd.DataFrame(all_params)
# tuning_results["mape"] = mapes
# best_params = tuning_results.sort_values("mape").iloc[0].to_dict()

# # Print the best parameters and time taken
# print("Best Parameters:", best_params)
# print("Time taken:", time.time() - start_time)













# df = pd.DataFrame({'dates': dates, 'values': values})
# df['dates'] = pd.to_datetime(df['dates'])
# df.set_index('dates', inplace=True)

# # Plot the time series
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['values'], label='Time Series Data')
# plt.xlabel('Date')
# plt.ylabel('Values')
# plt.title('Time Series Plot with Seasonality')
# plt.legend()
# plt.grid(True)
# plt.show()