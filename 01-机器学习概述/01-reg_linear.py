"""
## （一) 回归任务预测
# 1. 导入依赖包
# 2. 准备数据
# 3. 实例化线性回归模型
# 4. 模型训练
# 5. 模型预测

x = [[80, 86], [82, 80],[85, 78],[90, 90],[86, 82],[82, 90],[78, 80],[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

"""
# 1. 导入依赖包
from sklearn.linear_model import LinearRegression
import joblib

def dm01_regres():
    # 2. 准备数据
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
    # 3. 实例化线性回归模型
    regression = LinearRegression()
    # 4. 模型训练
    regression.fit(x, y)
    print(f'regression.coef_ --> {regression.coef_}')
    print(f'regression.intercept_ --> {regression.intercept_}')
    # 5. 模型预测
    print(regression.predict([[80, 86]]))
    # 6. 模型保存
    joblib.dump(regression, '../model/mymodel2.bin')
    # 7. 模型加载
    estimator1 = joblib.load('../model/mymodel2.bin')
    # 8. 模型再预测
    print(estimator1.predict([[90, 86]]))


if __name__ == '__main__':
    dm01_regres()
