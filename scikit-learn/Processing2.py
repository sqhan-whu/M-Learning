from sklearn.feature_selection import VarianceThreshold

###### As an example, suppose that we have a dataset with boolean features, 
###### and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples.
###### so we can select using the threshold .8 * (1 - .8):

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

print(sel.fit_transform(X))
## As expected, VarianceThreshold has removed the first column, which has a probability (5/6 >0.8)  of containing a zero.

##########################################################################################################################

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def datasets_demo():
	"""
	sklearn 数据集使用
	"""
	iris = load_iris()
	#print("yuanweihua shujuji :\n",iris)
	#print("--- :\n",iris.feature_names)
	#print("--- :\n",iris.data.shape)

	#数据集划分：
	x_train, x_test,y_train, y_test= train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
	#print(x_train,"\n\n",x_test)
	return x_train, x_test,y_train, y_test
  

x_train, x_test,y_train, y_test = datasets_demo()
X = x_train
"""
方差降维：
"""
sel = VarianceThreshold(threshold=(.8*(1- .8 )))
print(sel.fit_transform(X),sel.fit_transform(X).shape)



