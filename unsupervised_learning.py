import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from seaborn import clustermap
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from cart import cart_params

###################################
# K-Means
###################################

df = pd.read_csv('datasets/USArrests.csv' , index_col = 0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4 , random_state=17).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

########################################
# Optimum Küme Sayısının Belirlenmesi
########################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for i in K:
    kmeans = KMeans(n_clusters=i).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K değerlerine karşılık SSE/SSD/SSR")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k = (2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

#######################################
# Final Clusterların Oluşturulması
#######################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv('datasets/USArrests.csv' , index_col = 0)
df["cluster"] = clusters_kmeans

df.head()
df["cluster"] = df["cluster"] + 1

df.groupby("cluster").agg(["count" , "mean" , "median"])

df.to_csv('clusters.csv')


#################################
# Hierarchical Clustering
#################################

df = pd.read_csv('datasets/USArrests.csv' , index_col = 0)

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize = (10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size = 10)
plt.show()

plt.figure(figsize = (15, 10))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size = 10)
plt.show()


########################################
# Küme Sayısını Belirleme
########################################

plt.figure(figsize = (7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size = 10)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.show()

########################################
# Final Modelini Oluşturmak
########################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage='average')

clusters = cluster.fit_predict(df)

df = pd.read_csv('datasets/USArrests.csv' , index_col = 0)

df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1


########################################
# Principal Component Analysis
########################################

df = pd.read_csv('datasets/hitters.csv')
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and "Salary" not in col]

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

###################################
#Optimum Bileşen Sayısı
###################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


#####################################
# Final PCA Oluşturulması
#####################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

#####################################
# PCR : Principal Component Regression
#####################################

df = pd.read_csv('datasets/hitters.csv')
df.shape
len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]),
                      df[others]], axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
     label_encoder(final_df, col)

final_df.dropna(inplace = True)

y = final_df["Salary"]
X = final_df.drop("Salary", axis = 1)

lm = LinearRegression()

rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5,
                                        scoring='neg_mean_squared_error')))
y.mean()

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5,
                                        scoring='neg_mean_squared_error')))

cart_params = {'max_depth': range(1, 11),
               'min_samples_split': range(2, 20)}

cart_best_grid = GridSearchCV(cart, cart_params, cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)
cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_ , random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5,
                                        scoring='neg_mean_squared_error')))

#####################################
#Breast Cancer
#####################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('datasets/breast_cancer.csv')

y = df["diagnosis"]
X = df.drop("diagnosis", axis = 1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = pca_fit, columns =['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title(f'{target.capitalize()}', fontsize = 20)

    targets = list(dataframe[target].unique())

    colors = random.sample(['r', 'b', 'g', 'c', 'm', 'y', 'k'],  len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'],
                   dataframe.loc[indices, 'PC2'],
                   c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


#########################################
# Iris
########################################

import seaborn as sns
df = sns.load_dataset('iris')

y = df["species"]
X = df.drop("species", axis = 1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")

#########################################
# Diabetes
########################################

df = pd.read_csv('datasets/diabetes.csv')

y = df["Outcome"]
X = df.drop("Outcome", axis = 1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")