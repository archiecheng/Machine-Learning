# 1. 导入依赖包

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import  PCA
from sklearn.datasets import load_iris
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_iris
# 2.1 低方差过滤法
# data = pd.read_csv('./emails.csv')
# print(data.shape)
#
# transform = VarianceThreshold(threshold=0.1)
# x = transform.fit_transform(data)
# print(x.shape)

# 2.2 PCA主成分分析
# x, y = load_iris(return_X_y=True)
# print(x[:5])
# pca1 = PCA(n_components=0.95)
# print(pca1.fit_transform(x))
# pca1 = PCA(n_components=3)
# print(pca1.fit_transform(x))

# # 2.3 相关系数法
data = load_iris()
data = pd.DataFrame(data.data, columns=data.feature_names)
print(data)
# 3. 皮尔逊相关系数
corr = pearsonr(data['sepal length (cm)'], data['sepal width (cm)'])
print(corr, '皮尔逊相关系数:', corr[0],'不相关性概率:', corr[1])

# 4. 斯皮尔曼相关系数
corr = spearmanr(data['sepal length (cm)'], data['sepal width (cm)'])
print(corr, '斯皮尔曼相关系数:', corr[0],'不相关性概率:', corr[1])