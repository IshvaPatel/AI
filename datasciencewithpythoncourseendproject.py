# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("311_Service_Requests_from_2010_to_Present.csv")

# %%

data.describe()


# %%
data.columns


# %%
data.shape


# %%
data.isna().sum()

# %%
x=np.ndarray(53)
a=0
for i in data.columns:
    x[a]=data[i].isna().sum()
    a=a+1
plt.bar(data.columns,x)
plt.xticks(rotation=90)
plt.show()

# %%
data.dropna(subset=['Closed Date'],inplace=True)
data["Closed Date"].isna().sum()

# %%

data['dateCreated']=pd.to_datetime(data["Created Date"])
data['dateClosed']=pd.to_datetime(data["Closed Date"])
data['elapsed']=(data["dateClosed"]-data["dateCreated"]).dt.total_seconds()
data

# %%
data['elapsed'].describe()

# %%
data[["City","Complaint Type"]].isna().sum()

# %%
## 2.3
from sklearn.impute import SimpleImputer
imp_values=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='Unknown City')
imp_values.fit(data[["City","Unique Key"]])
SimpleImputer(strategy='constant',fill_value='Unknown City')
data["City"]=imp_values.transform(data[["City","Unique Key"]])
data[["City","Complaint Type"]].isna().sum()

# %%
import seaborn as sns
sns.countplot(x='City',data=data)
plt.xticks(rotation=90)
plt.show()


# %%
data.groupby("City").get_group("BROOKLYN")[["Longitude","Latitude"]].plot(kind="scatter",x='Longitude',y='Latitude')


# %%
data.groupby("City").get_group("BROOKLYN")[["Longitude","Latitude"]].plot(kind="hexbin",x='Longitude',y='Latitude',colormap='jet')


# %%
import seaborn as sns
sns.countplot(x='Complaint Type',data=data)
plt.xticks(rotation=90)
plt.show()

# %%
sns.countplot(x='Complaint Type',data=data.groupby('City').get_group('NEW YORK'))
plt.xticks(rotation=90)
plt.show()

# %%
data.groupby("City").get_group('NEW YORK')["Complaint Type"].value_counts().head(10)

# %%
data.groupby('City')["Complaint Type"].value_counts()

# %%
df_new=data.pivot_table(columns=data["City"], index=data["Complaint Type"],values= 'Unique Key',aggfunc='count')
df_new

# %%
d=df_new.transpose()
d.plot(kind='bar',figsize=(20,20))

# %%
df=data[['City','Complaint Type','elapsed']]
df

# %%
df.groupby(["City","Complaint Type"])['elapsed'].mean()

# %%
df.groupby(["City","Complaint Type"])['elapsed'].mean().plot(kind='bar',fontsize='0.1')

# %%
df.groupby(["Complaint Type"])['elapsed'].mean().plot(kind='bar')

# %%
from scipy.stats import kruskal

import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests

d = data[['elapsed','Complaint Type']]
groups = pd.unique(d['Complaint Type'])
d1 = {grp:d['elapsed'][d['Complaint Type'] == grp]  for grp in groups}
stat, p_value = kruskal(*d1)
stat,p_value

# %%
alpha = 0.05
if p_value < alpha:
    print('Reject Null Hypothesis (Different distribution)')
else:
    print('Do not Reject Null Hypothesis (Same distribution)')


