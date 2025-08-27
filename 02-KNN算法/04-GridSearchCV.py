# 1 导入依赖包
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 2 加载数据集并展示数据集
# 2.1 加载数据集
iris_data = load_iris()

# print(iris_data.keys()) # --> 输出 data 里的数据字段名称

# print("iris_data.data\n")
# print(iris_data.data[:5])
# print("---------------\n")
# print("iris_data.target\n")
# print(iris_data.target)
# print("---------------\n")
# print("iris_data.target_names\n")
# print(iris_data.target_names)
# print("---------------\n")
# print("iris_data.feature_names\n")
# print(iris_data.feature_names)


# 2.2 展示数据集
# 把数据转换为 dataframe 格式 设置 data, columns 属性, 目标值名称
iris_df = pd.DataFrame(iris_data['data'], columns=iris_data.feature_names)
# print(iris_df)
iris_df['label'] = iris_data.target
# print(iris_df)
sns.lmplot(x = 'sepal length (cm)', y = 'petal width (cm)', data = iris_df, hue = 'label', fit_reg=False)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

# 3 特征工程(预处理-标准化)
## 3.1 数据集划分
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, train_size=0.8, random_state=1)
print(f'len(X_train) -->{len(X_train)}')
print(f'len(X_test) -->{len(X_test)}')
# print(f'len(y_train) -->{len(y_train)}')
# print(f'len(y_test) -->{len(y_test)}')
print(f"len(iris_data.data) --> {len(iris_data.data)}")
## 3.2 标准化

# fit -- 计算规律
# fit_transform -- 计算规律 + 数据转换
# transform -- 无需计算, 根据之前计算的规律, 进行转化
# estimator.predict() 直接预测结果
# estimator.predict_proba() 预测概率
pre = StandardScaler()
x_train = pre.fit_transform(X_train)
x_test = pre.fit_transform(X_test)

#  4. 模型训练
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
print(model.predict(x))




















