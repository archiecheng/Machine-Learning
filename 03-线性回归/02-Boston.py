# 1 导入库
from sklearn.preprocessing import StandardScaler # 特征处理
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, r2_score          # 均方误差评估
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV
from fairlearn.datasets import fetch_boston

import pandas as pd
import numpy as np


# 数据预处理
# 获取数据
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
# print(f'raw_df.head():-->\n{raw_df.head()}')
# data = np.hstack(raw_df.values[::2,:], raw_df.values[1::2, :2])
# print(f'data --> \n{data}')
# target = raw_df.values[1::2,2]
# 直接获取数据
boston = fetch_boston()
X = boston['data']
y = boston['target']
columns = boston['feature_names']

# 转成 DataFrame（便于查看）
df = pd.DataFrame(X, columns=columns)
df['MEDV'] = y

print("前5行数据：\n", df.head())

# 数据集划分（注意这里用 X, y 而不是 data, target）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=22
)

# 用 Pipeline：标准化 + 线性回归（SGDRegressor）
model = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1000,
        tol=1e-3,
        random_state=22
    )
)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print("预测前10个：", np.round(y_pred[:10], 2))

# 评估
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))











