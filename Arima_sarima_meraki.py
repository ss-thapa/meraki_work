import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
pd.set_option('display.max_columns', None)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



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


log_shift = daily_data[['grand_total']].copy(deep=True)
log_shift['log'] = np.log(log_shift['grand_total'])
log_shift['logShift'] = log_shift['log'].shift()
log_shift['logShiftDiff'] = log_shift['log'] - log_shift['logShift']
# print(log_shift.head())


# In[25]:


# test_stationarity(log_shift.dropna(),'logShiftDiff')



#### arima model building 


# plot_pacf(daily_data['grand_total'],lags=20)


# plot_acf(daily_data['grand_total'])
# plt.show()


p = 1
d = 0
q = 3


train_data = daily_data[daily_data.index < '2024-01-15']

test_data = daily_data[daily_data.index >= '2024-01-15']

test_data = test_data.drop('2024-02-03')



model = ARIMA(test_data['grand_total'], order= (1,0,3))

model_fit = model.fit()

print(model_fit)
# prediction = model_fit.predict(start = test_data.index[0], end = test_data.index[-1])

# print(prediction)

