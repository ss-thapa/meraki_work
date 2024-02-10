import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from prophet import Prophet



pd.set_option('display.max_columns', None)



df_sales_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_main.csv')
df_sales_detail = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/sales_detail.csv')
# df_purchas_main = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/purchase_main.csv')
# df_purchase_detail = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/purchase_detail.csv')

df_item = pd.read_csv('/Users/sunilthapa/Desktop/My_projects/meraki/datas/items.csv')



# print(df_sales_main[df_sales_main['grand_total'] > 30000])





# # print(df_item[df_item['id']== 787])

# # print(df_sales_main[df_sales_main['sub_total'] > 10000].sample(5))

# print(df_sales_detail[df_sales_detail['gross_amount'] > 10000].head())



col_name =['created_date_ad','grand_total']

final_df = df_sales_main[col_name]

final_df['created_date_ad'] = pd.to_datetime(final_df['created_date_ad'])


final_df =final_df[final_df['grand_total'] < 20000]



# plt.figure(figsize=(15,6))
# plt.plot(final_df['created_date_ad'],final_df['grand_total'])
# plt.xticks(rotation='vertical')
# plt.show()

final_df =final_df.rename(columns={'created_date_ad':'ds','net_amount':'y'})

final_df['ds'] = final_df['ds'].dt.tz_localize(None)


print(final_df.tail())



# mod = Prophet()

# model = mod.fit(final_df)

# future = mod.make_future_dataframe(periods=7,freq='D')

# forecast = mod.predict(future)

# print(forecast.head())


