from sklearn.feature_extraction import DictVectorizer

def dict_demo():
	"""
	字典特征提取
	"""
	data= [{'city':'BJ','t':100},{'city':'SH','t':60},{'city':'SZ','t':30}]
	transfer = DictVectorizer(sparse=False)
	data_new = transfer.fit_transform(data)
	print (data_new)
	print(transfer.get_feature_names())
dict_demo()

from sklearn.feature_extraction.text import CountVectorizer

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
count_demo()


