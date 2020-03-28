from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def dict_demo():
	"""
	字典特征提取
	"""
	data= [{'city':'BJ','t':100},{'city':'SH','t':60},{'city':'SZ','t':30}]
	transfer = DictVectorizer(sparse=False)
	data_new = transfer.fit_transform(data)
	print (data_new)
	print(transfer.get_feature_names())


def count_demo():
	"""
	文本特征提取
	"""
	data = ["life is short, i like like python", "life is too long, i dislike python"]

	transfer = CountVectorizer()

	data_new = transfer.fit_transform(data)
	print("data_new:\n",data_new.toarray())
	print(transfer.get_feature_names())

	return None
 
def minmax_demo():
	"""
	数据归一化
	"""
	data = pd.read_table("datingTestSet.txt")
	"""
	M		L			C			T
	40920	8.326976	0.953952	largeDoses
	14488	7.153469	1.673904	smallDoses
	26052	1.441871	0.805124	didntLike
	"""

	data.loc[data['T'] == 'largeDoses','T'] = 3
	data.loc[data['T'] == 'smallDoses','T'] = 2
	data.loc[data['T'] == 'didntLike','T'] = 1
	
	data = data.iloc[:,:3]
	
	transfer = MinMaxScaler()
	data_new = transfer.fit_transform(data)
	print(data_new)

	return None

#minmax_demo()

def stand_demo():
	"""
	标准化
	"""
	data = pd.read_table("datingTestSet.txt")
	data.loc[data['T'] == 'largeDoses','T'] = 3
	data.loc[data['T'] == 'smallDoses','T'] = 2
	data.loc[data['T'] == 'didntLike','T'] = 1
	
	data = data.iloc[:,:3]

	transfer = StandardScaler()
	data_new = transfer.fit_transform(data)
	print(data_new)

stand_demo()
