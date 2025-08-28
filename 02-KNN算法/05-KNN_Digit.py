"""
## KNN 手写数字识别
# 1. 导入依赖包
# 2. 读取数据并展示数据
# 2.1 读取数据
# 2.2 展示数据
# 3. 数据预处理
# 3.1 归一化
# 3.2 数据集划分
# 4. 模型训练
# 4.1 实例化模型
# 4.2 模型训练
# 5. 模型预测
# 6. 模型评估
# 7. 模型保存
# 8. 模型加载
"""

# 1. 导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

import warnings
warnings.filterwarnings('ignore')
# 2. 读取数据并展示数据
# 2.1 读取数据
data = pd.read_csv('train.csv')
x = data.iloc[:,1:] # iloc 通过位置索引来获取数据
y = data.iloc[:,0]
# 2.2 展示数据
digit = x.iloc[1000].values
img = digit.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.imsave('demo.png', img)
plt.show()
# 3. 数据预处理
# 3.1 归一化
x = x / 255. # 浮点数
# 3.2 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)
# 4. 模型训练
# 4.1 实例化模型
model = KNeighborsClassifier(n_neighbors=11)
# 4.2 网格搜索 + CV
model = GridSearchCV(model, param_grid={'n_neighbors': [1, 3, 5, 7, 9, 11]}, cv=5)
# 4.3 模型训练
model.fit(x_train, y_train)
print(model.best_estimator_)
# 5. 模型预测
img = plt.imread('demo.png')
img = img.reshape(1, -1)
y_predict = model.best_estimator_.predict(x_test)
print(y_predict)
# 6. 模型评估
print(model.score(x_test, y_test))
print(accuracy_score(y_predict, y_test))
# 7. 模型保存
joblib.dump(model, 'knn.pth')
# 8. 模型加载
knn = joblib.load('knn.pth')
print(knn.score(x_test, y_test))
img = plt.imread('demo.png')
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
img = img[:,:,1].reshape(1, -1)
print(knn.predict(img))