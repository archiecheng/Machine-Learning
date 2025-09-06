"""
Adaboost
1. 导入依赖包
2. 读取数据以及数据预处理
2.1 读取数据
2.2 数据预处理
2.3 数据集划分
3. 模型训练
4. 模型评估
"""


# 1. 导入依赖包
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 2. 读取数据以及数据预处理
# 2.1 读取数据
data = pd.read_csv('./wine0501.csv')
print(data.info)

# 2.2 数据预处理
data = data[data['Class label'] != 1]
x = data[['Alcohol', 'Hue']].copy()
y = data['Class label'].copy()
print(y)
pre = LabelEncoder()
y = pre.fit_transform(y)
print(y)

# 2.3 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x,y)
# 3. 模型训练
dt = DecisionTreeClassifier(criterion='entropy', max_depth=1)
dt.fit(x_train, y_train)
ada = AdaBoostClassifier(estimator=dt, learning_rate=0.1, n_estimators=50)
ada.fit(x_train, y_train)
# 4. 模型评估
print(ada.score(x_test,y_test))