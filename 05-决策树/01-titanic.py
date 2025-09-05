"""
## 泰坦尼克号幸存预测
# 1.导入依赖包
# 2.读取数据及预处理
# 2.1 读取数据
# 2.2 数据处理
# 3 模型训练
# 4 模型预测与评估
# 4.1 模型预测
# 4.2 模型评估
# 4.3 可视化

"""
# 1. 导入依赖包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 2.读取数据及预处理
# 2.1 读取数据
data = pd.read_csv('./titanic/train.csv')
print(f'data.head()-->{data.head()}')
print(f'data.info-->{data.info}')
print(f'data.columns-->{data.columns}')
# 2.2 数据处理
x = data[['Pclass','Sex','Age']].copy()
y = data['Survived'].copy()
print(x.head(10))


x['Age'].fillna(x['Age'].mean(), inplace=True)
print(x.head(10))
x = pd.get_dummies(x)
print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# 3. 模型训练
model = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)

model.fit(x_train, y_train)
# 4 模型预测与评估
# 4.1 模型预测
y_pred = model.predict(x_test)
# 4.2 模型评估
print(classification_report(y_true=y_test, y_pred=y_pred))
# 4.3 可视化
plot_tree(model,filled=True, feature_names=['Pclass','Age','Sex_female','Sex_male'],class_names=['died','Survived'])
plt.show()


