# 欠拟合模拟

# # 1 导入工具包
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
#
# # 2 准备数据
# np.random.seed(22)
# x = np.random.uniform(-3, 3, size=100)
# print(x)
# y = 0.5 * x ** 2 + 2 + np.random.normal(0,1,size=100)
# print(y)
#
# # 3 模型训练
# model = LinearRegression()
# x = x.reshape(-1,1)
# model.fit(x,y)
#
# # 4 模型预测
# # 4-1 模型预测
# y_predict = model.predict(x)
# print(mean_squared_error(y,y_predict))
#
# # 4-2 展示效果
# plt.scatter(x,y)
# plt.plot(x,y_predict)
# plt.show()


# 正好拟合

# 1 导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2 准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + 2 + np.random.normal(0,1,size=100)
# print(y)

# 3 模型训练
model = LinearRegression()
X = x.reshape(-1,1)
X2 = np.hstack([X, X ** 2])
model.fit(X2, y)

# # 4 模型预测
# # 4-1 模型预测
y_predict = model.predict(X2)
print(mean_squared_error(y_true=y,y_pred=y_predict))

# # 4-2 展示效果
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort((x))])
plt.show()
