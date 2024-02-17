import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = sns.load_dataset('flights')
df['yearMonth'] = pd.to_datetime("01-"+df['month'].astype(str)+"-"+df['year'].astype(str))
df.set_index('yearMonth',inplace=True)



plt.figure(figsize=(10,8))
sns.lineplot(data=df,x=df.index,y=df['passengers'])
plt.show()



