#!/usr/bin/env python
# coding: utf-8

import pandas as pd


#获取数据
titanic = pd.read_csv("D:/BaiduNetdiskDownload/机器学习/机器学习day2资料/02-代码/titanic.csv")

x = titanic[["pclass","age","sex"]]
y = titanic["survived"]

#缺失值处理
x["age"].fillna(x["age"].mean(),inplace=True)


#转换成字典
x = x.to_dict(orient="records")


#数据集划分
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22)


#字典特征抽取
from sklearn.feature_extraction import DictVectorizer
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#决策树预估器：
from sklearn.tree import DecisionTreeClassifier, export_graphviz

estimator = DecisionTreeClassifier(criterion="entropy",max_depth=8)
estimator.fit(x_train,y_train)
#模型评估：
y_predict = estimator.predict(x_test)
print("y_predict:\n",y_predict)
print("直接比对真实值和预测值：\n",y_test == y_predict)
score = estimator.score(x_test,y_test)
print("准确率：\n",score)
#可视化决策树 ## http://webgraphviz.com/
export_graphviz(estimator,out_file="titanic.dot",feature_names=transfer.get_feature_names())
