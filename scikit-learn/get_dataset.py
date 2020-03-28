from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def datasets_demo():
	"""
	sklearn 数据集使用
	"""
	iris = load_iris()
	#print("yuanweihua shujuji :\n",iris)
	print("--- :\n",iris.feature_names)
	print("--- :\n",iris.data.shape)

	#数据集划分：
	x_train, x_test,y_train, y_test= train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
	print(len(y_train),"\n",len(y_test))
	return None


datasets_demo()
