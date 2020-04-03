### instacrat Market

import pandas as pd
order_products = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")
aisles = pd.read_csv("aisles.csv")

table1 = pd.merge(orders,order_products,on="order_id")
table2 = pd.merge(table1,products,on="product_id")
table3 = pd.merge(table2,aisles,on="aisle_id" )

table = pd.crosstab(table3["user_id"],table3["aisle_id"])
data = table[:10000]
#print(table.head())
#
from sklearn.decomposition import PCA
#1. 实例一个转化器类型：
transfer = PCA(n_components=0.95)
#2. 调用fit_transform
data_new = transfer.fit_transform(data)

print(data_new.shape)


##KMeans 聚类
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=3)
estimator.fit(data_new)
y_predict = estimator.predict(data_new)
from sklearn.metrics import silhouette_score
silhouette_score(data_new,y_predict)
