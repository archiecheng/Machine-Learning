# 1 导入依赖包
from sklearn.linear_model import LinearRegression
# 2 准备数据身高和体重
x = [[160], [166], [172], [174], [180]]
y = [56.3, 60.6, 65.1, 68.5, 75]
# 3 实例化线性回归模型 estimator
estimator = LinearRegression()
# 4 训练线性回归模型 fit() h(w) = w1x1 + w2x2 +b
estimator.fit(x, y)
# 打印线性回归模型参数 coef_ intercept_
print("estimator.coef_ -->", estimator.coef_)
print("estimator.intercept_-->",estimator.intercept_)
# 5 模型预测 predict()
myres = estimator.predict([[176]])
print("myres --> ", myres)