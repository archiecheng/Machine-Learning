"""
## GBDT 梯度提升树
# 1. 导入依赖包
# 2. 读取数据及数据预处理
# 2.1 读取数据
# 2.2 数据预处理
# 2.3 划分数据集
# 3. 模型训练
# 4. 模型评估
"""

# 1. 导入依赖包
import pandas as pd
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 2. 读取数据及数据预处理
# 2.1 读取数据
data = pd.read_csv('./train.csv')
print(data.head())
print(data.info)
# 2.2 数据预处理
x = data[['Pclass', 'Sex', 'Age']].copy()
y = data['Survived'].copy()
print(x.head(0))
# 填充缺失值
x['Age'].fillna(x['Age'].mean(), inplace=True)
print(x.head(10))
x = pd.get_dummies(x)
print(x.head(10))
# 2.3 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3. 模型训练
model = GradientBoostingClassifier()
# # 交叉验证和网格搜索
model_GBDT = GridSearchCV(model, param_grid={'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                                'max_depth': [1, 3, 4, 5, 7], 'random_state': [0]}, cv=4)

model_GBDT.fit(x_train, y_train)
# 4. 模型评估
print(model.score(x_test, y_test))
