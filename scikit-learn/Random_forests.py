import pandas as pd
titanic = pd.read_csv("D:/BaiduNetdiskDownload/机器学习/机器学习day2资料/02-代码/titanic.csv")
x = titanic[['pclass','age','sex']]
y = titanic['survived']
#缺省值填补
x['age'].fillna(x['age'].mean(),inplace=True)

x = x.to_dict(orient="records")
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22)

from sklearn.feature_extraction import DictVectorizer
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV

estimator = RandomForestClassifier()
param_dict = {"n_estimators":[80,100,120,140,160]}
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)

estimator.fit(x_train,y_train)

y_predict = estimator.predict(x_test)
print("直接比对真实值和预测值：\n",y_test == y_predict)
score = estimator.score(x_test,y_test)
print("准确率：\n",score)
print("最佳参数：\n", estimator.best_params_)
print("最佳结果：\n", estimator.best_score_)
print("最佳估计器:\n", estimator.best_estimator_)
