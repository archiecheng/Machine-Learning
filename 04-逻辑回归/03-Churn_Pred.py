# 1 导入依赖包
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# 数据处理
data = pd.read_csv('churn.csv')
print(f'data.info-->{data.info}')
print(f'data.head()-->{data.head()}')
print(f'data.describe()-->{data.describe()}')


# 处理类别型的数据 类别型数据做 one-hot 编码
data = pd.get_dummies(data)
print(f"One-hot编码结果-->\n{data.head()}")

# 删除两列
data = data.drop(['Churn_No','gender_Male'],axis=1)
print(f"删除两列后的结果-->\n{data.head()}")


data = data.rename(columns={'Churn_Yes':'flag'})
print(data.head())
print(data.flag.value_counts())


# 3. 特征工程
sns.countplot(data=data, y='PaymentElectronic',hue='flag')
plt.show()

x = data[['PaymentElectronic', 'Contract_Month', 'internet_other']]
y = data['flag']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=100)

# 4. 模型训练
LR = LogisticRegression()
LR.fit(x_train, y_train)

# 5. 模型评估
y_predict = LR.predict(x_test)
print(f'ACC:\n{accuracy_score(y_test,y_predict)}')
print(f'ROC_AUC:\n{roc_auc_score(y_test,y_predict)}')
print(f'分类报告:\n{classification_report(y_test,y_predict)}')











