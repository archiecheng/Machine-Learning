# 1 导入依赖包
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 1 加载数据
iris_data = load_iris()

# 2 数据集划分
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, train_size=0.8, random_state=1)

# 3 特征预处理
pre = StandardScaler()
x_train = pre.fit_transform(X_train)
x_test = pre.fit_transform(X_test)

# 4 模型实例化 + 交叉验证 + 网格搜索
model = KNeighborsClassifier(n_neighbors=1)
estimator = GridSearchCV(model, param_grid={'n_neighbors': [4, 5, 7, 9]}, cv=4)
estimator.fit(x_train, y_train)

print(estimator.best_score_)
# print(estimator.best_estimator_)
# print(estimator.cv_results_)

pd.DataFrame(estimator.cv_results_).to_csv('./grid_search.csv')

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
x = [[5.1, 3.5, 1.4, 0.2]]
x = pre.transform(x)
y_prdict = model.predict(x_test)
print(accuracy_score(y_test, y_prdict))




















