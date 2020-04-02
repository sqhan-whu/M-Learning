#!/usr/bin/env python
# coding: utf-8

import pandas as pd

data = pd.read_csv("D:/BaiduNetdiskDownload/机器学习/机器学习day2资料/02-代码/FBlocation/train.csv")

#缩小数据范围：
data = data.query("x<2.5 & x >2 & y<1.5 &y >1")

#处理时间特征：
time_value = pd.to_datetime(data['time'],unit='s')

date = pd.DatetimeIndex(time_value)

data["day"]=date.day

data["hour"]=date.hour

data["weekday"]=date.weekday

place_count = data.groupby("place_id").count()["row_id"]


place_count[place_count>3].head().index.values

data_final = data[data["place_id"].isin(place_count[place_count>3].index.values)]

#筛选特征值和目标值：
x = data_final[["x","y","accuracy","day","weekday","hour"]]

y = data_final["place_id"]


#数据集划分：
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test=train_test_split(x,y)


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = KNeighborsClassifier()

param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
estimator.fit(x_train, y_train)

y_predict = estimator.predict(x_test)


print("y_predict:\n", y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 最佳参数：best_params_
print("最佳参数：\n", estimator.best_params_)
 # 最佳结果：best_score_
print("最佳结果：\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
