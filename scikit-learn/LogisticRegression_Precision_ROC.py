import pandas as pd
import numpy as np
path = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_name = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv(path,names=column_name)
#替换缺失值
data = data.replace(to_replace='?',value=np.nan)
data.dropna(inplace=True)
data.isnull().any() #所有列不存在缺失值

from sklearn.model_selection import train_test_split
x = data.iloc[:,1:-1]
y = data["Class"]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=22)

#标准化：
from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#逻辑回归：
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(x_train,y_train)

#逻辑回归模型参数：
#得出模型
print("正规方程-权重系数为：\n", estimator.coef_)
print("正规方程-偏置为：\n", estimator.intercept_)

#模型评估：
y_predict = estimator.predict(x_test)
print("y_predict:\n",y_predict)
print("直接比对真实值和预测值：\n",y_test==y_predict)
score= estimator.score(x_test,y_test)
print("准确率:\n",score)

#精准率和召回率计算
from sklearn.metrics import classification_report
report = classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶行"])
print(report)

#ROC曲线，AUC 面积
from sklearn.metrics import roc_auc_score
# y_true ：每个样本的真实类别，必须是0（反例），1（正例）
#将t_test 转换成0 1 
y_true= np.where(y_test > 3, 1,0)
auc = roc_auc_score(y_true, y_predict)
print(auc)

