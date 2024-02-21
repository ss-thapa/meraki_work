import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
pd.set_option('display.max_columns', None)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import itertools



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')


col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])




final_df['created_date_ad'] = final_df['created_date_ad'].dt.tz_localize(None)

daily_data = final_df.resample('D', on='created_date_ad').sum().reset_index()

daily_data = daily_data.set_index('created_date_ad')


##dynamic testing of stationarity

def test_stationarity(dataFrame, var):
    dataFrame['rollMean']  = dataFrame[var].rolling(window=12).mean()
    dataFrame['rollStd']  = dataFrame[var].rolling(window=12).std()
    
    from statsmodels.tsa.stattools import adfuller
    adfTest = adfuller(dataFrame[var],autolag='AIC')
    stats = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    print(stats)
    
    for key, values in adfTest[4].items():
        print('criticality',key,":",values)
        
    sns.lineplot(data=dataFrame,x=dataFrame.index,y=var)
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollMean')
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollStd')
    plt.show()






#### arima model building 


# plot_pacf(daily_data['grand_total'],lags=20)


# plot_acf(daily_data['grand_total'])
# plt.show()




train_data = daily_data[daily_data.index < '2024-01-15']

test_data = daily_data[daily_data.index >= '2024-01-15']

test_data = test_data.drop('2024-02-03')

train_data = train_data[['grand_total']].copy(deep=True)
train_data['log'] = np.log(train_data['grand_total'])
train_data['logShift'] = train_data['log'].shift()
train_data['logShiftDiff'] = train_data['log'] - train_data['logShift']




# print(train_data)

# p=d=q= range()
# pdq = list(itertools.product(p,d,q))

# for param in pdq:
#     try:
#         model_arima = ARIMA(train_data['logShiftDiff'],order=param)
#         model_arima_fit = model_arima.fit()
#         print(param,model_arima_fit.aic)
#     except:
#         continue




# plot_acf(train_data['logShiftDiff'])
# plt.title('Autocorrelation Function (ACF)')
# plt.show()

# plot_pacf(train_data['logShiftDiff'], lags=50)
# plt.title('Partial Autocorrelation Function (PACF)')
# plt.show()






model = ARIMA(train_data['logShift'],order=(20,1,5))

model_result = model.fit()


prediction = model_result.predict(start = test_data.index[0], end = test_data.index[-1],dynamic=False)
y_pred_original_scale = np.exp(prediction)


# mape = mean_absolute_percentage_error(test_data['grand_total'],y_pred_original_scale)

# print(mape)


