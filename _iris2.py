import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
iris_dataset =load_iris()
from sklearn.model_selection import train_test_split
#随机打乱数据集
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
iris_dataset['target'], random_state=0)
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, 
figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
#plt.show()

#KNN算法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

#打印结果
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))