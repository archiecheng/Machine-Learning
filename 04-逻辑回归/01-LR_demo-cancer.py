# 1. 导入依赖包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 2. 加载数据以及数据预处理
data = pd.read_csv('./breast_cancer_bd.csv')
data.info()
# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()
# 2.2 确定特征值, 目标值
x = data.iloc[:,1:-1]
print('x.head()-->\n', x.head())
y = data["Class"]
print('y.head()-->\n',y.head())

# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=22)

# 3. 特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练, 机器学习(逻辑回归)
estimator = LogisticRegression()
estimator.fit(x_train,y_train)

# 5. 模型预测和评估
y_predict = estimator.predict(x_test)
print('y_predict-->',y_predict)
accuracy = estimator.score(x_test, y_test)
print('accuracy-->', accuracy)
print(estimator.score(x_test,y_predict))
