# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# %%
data=pd.read_excel('1673873388_rolling_stones_spotify.xlsx')
data

# %%
data.describe()

# %%
data.shape

# %%
data.info()

# %%
data.isna().sum()

# %%
np.array(data.columns.duplicated())

# %%
data.drop_duplicates(inplace=True)

# %%
data.shape

# %%
data['duration_ms']=data['duration_ms'].apply(lambda x: round(x/1000))

data

# %%
from datetime import date
today=date.today()

data['elapsed(yrs)']=today-data['release_date'].dt.date
data['elapsed(yrs)']=data['elapsed(yrs)'].dt.days
data['elapsed(yrs)']=(data['elapsed(yrs)']).apply(lambda x: round(x/365))
data['elapsed(yrs)']

# %%
plt.figure(figsize=(20,20))
data.boxplot(column=['acousticness','danceability','energy','instrumentalness','liveness','loudness'])

# %%

plt.figure(figsize=(20,20))
data.boxplot(column=['speechiness','tempo','valence','popularity','duration_ms'])

# %%
def detect_outliers_iqr(data):
    outlier_list = []
    data = sorted(data)
    q1 = np.percentile(data,25)
    q3 = np.percentile(data, 75)
    #print("The Val of Q1 and Q2",q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    #print("The lower & Upper Bound",lwr_bound, upr_bound)
    
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outlier_list.append(i)
    return outlier_list # Driver code



for i in ['loudness','instrumentalness','speechiness','tempo','popularity']:
    outliers = detect_outliers_iqr(data[i])
    print("Outliers in",i,"attribute :", outliers)

# %%
def handle_outliers(data):

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    IQR = q3-q1

    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)

    b = np.where(data<lwr_bound, lwr_bound, data)

    b1 = np.where(b>upr_bound, upr_bound, b)
    return b1
    
    
for i in ['acousticness','energy','loudness','tempo','popularity']:
    data[i]=handle_outliers(data[i])

# %%
plt.figure(figsize=(20,20))
data.boxplot(column=['acousticness','danceability','energy','instrumentalness','liveness','loudness'])

# %%
plt.figure(figsize=(20,20))
data.boxplot(column=['speechiness','tempo','valence','popularity','duration_ms'])

# %%
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler1=StandardScaler()
scaler2=MinMaxScaler()
data[['speechiness','duration_ms','instrumentalness','loudness','tempo','popularity']]=scaler2.fit_transform(data[['speechiness','duration_ms','instrumentalness','loudness','tempo','popularity']])

# %%
data.describe()

# %%
df=data.drop(columns=['Unnamed: 0','track_number','id','uri','release_date','name','album'],axis=1)
df

# %%
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)

# %%
sns.regplot(data=df,x='loudness',y='energy')


# %%
sns.regplot(data=df,x='valence',y='danceability')

# %%
sns.regplot(data=df,x='liveness',y='energy')

# %%

data.groupby('album')['popularity'].mean().plot(kind='bar',fontsize=5)
#sticky fingers remastered has the highest popularity,followed by some girls,exile on the mainstreet(2010 remastered),Tattoo You(2009 Remastered)

# %%
data.groupby('album')['popularity'].mean().sort_values(ascending=False)

# %%
d=data.groupby('album').get_group('Sticky Fingers (Remastered)')
d.plot(kind='bar',x='name',y='popularity')
## all the songs of the highest popular album are very popular

# %%
d=data.groupby('album').get_group("England's Newest Hit Makers")
d.plot(kind='bar',x='name',y='popularity')
## no songs of the least popular album are a hit

# %%
sns.regplot(data=df,x='acousticness',y='popularity')

# %%
sns.regplot(data=df,x='danceability',y='popularity')

# %%
sns.regplot(data=df,x='speechiness',y='popularity')

# %%
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(df)
pc=pca.explained_variance_ratio_
pc

# %%
PC_values = np.arange(pca.n_components) + 1
plt.plot(PC_values, pc, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# %%
pca2=PCA(n_components=2)
pca_result=pca2.fit_transform(df)
pd.DataFrame(pca_result)

# %%
pca2.components_

# %%
dataset_pca = pd.DataFrame(abs(pca2.components_), columns=df.columns, index=['PC_1', 'PC_2'])
dataset_pca

# %%
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(pca_result[:, 0], pca_result[:, 1])

ax.set_title('PCA - 2 dimensions')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')

plt.show()

# %%
from sklearn.cluster import KMeans


# %%
wcss=[]      #within a cluster sum of square (c-xi)2         
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=1) 
    kmeans.fit_predict(pca_result)
    wcss.append(kmeans.inertia_)  

# %%
plt.plot(range(1,11),wcss,"*--")
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# %%
from sklearn.metrics import silhouette_score
range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
silhouette_scores=[]
for n_clusters in range_n_clusters:
    my_cluster_model = KMeans(n_clusters=n_clusters)
    m = my_cluster_model.fit_predict(pca_result)
    silhouette_avg = silhouette_score(pca_result, m)
    silhouette_scores += [silhouette_avg]  
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# %%
plt.plot(range_n_clusters, silhouette_scores)

plt.title('Silhouette Score', fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

# %%
kmeans=KMeans(n_clusters=2)
kmeans.fit_predict(pca_result)
df['cluster'] = kmeans.labels_
data['cluster'] = kmeans.labels_

# %%
df['cluster'].value_counts()

# %%
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_)

ax.set_title('PCA - 2 dimensions')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')

plt.show()

# %%
sns.pairplot(df,kind='scatter',hue='cluster')

# %%
sns.lmplot(data,x='elapsed(yrs)',y= 'popularity',hue='cluster')

# %%
sns.lmplot(data,x='liveness',y= 'elapsed(yrs)',hue='cluster')

# %%
sns.lmplot(data,x='loudness',y= 'liveness',hue='cluster')

# %%
sns.lmplot(data,x='loudness',y= 'energy',hue='cluster')

# %%
sns.lmplot(data,x='valence',y= 'danceability',hue='cluster')

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data['loudness'], data['energy'], data['popularity'], c=data['cluster'])
ax.set_xlabel('loudness')
ax.set_ylabel('energy')
ax.set_zlabel('popularity')

# %%



