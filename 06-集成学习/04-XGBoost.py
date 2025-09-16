# # 1. 导入依赖包
# import joblib
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# from collections import Counter
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils import class_weight
#
# # 2. 数据读取及数据预处理
# # 2.1 数据获取
# data = pd.read_csv('./winequality-red.csv')
# # print(data)
#
# # 2.2 数据预处理
# x = data.iloc[:,:-1]
# y = data.iloc[:,-1]
#
# # 2.3 数据集划分
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)
#
# # 2.4 数据存储
# pd.concat([x_train, y_train], axis=1).to_csv('./红酒品质分类_train.csv')
# pd.concat([x_test, y_test], axis=1).to_csv('./红酒品质分类_test.csv')
#
# # 2. 数据读取及数据预处理
# # 2.1 数据获取
# train_data = pd.read_csv('./红酒品质分类_train.csv')
# test_data = pd.read_csv('./红酒品质分类_test.csv')
# # 2.2 数据预处理
# x_train = train_data.iloc[:,:-1]
# y_train = train_data.iloc[:,-1]
# x_test = test_data.iloc[:,:-1]
# y_test = test_data.iloc[:,-1]
# # 样本均衡化
# class_weight = class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
#
# # 3. 模型训练
# model = XGBClassifier(n_estimators=5, objective='multi:softmax')
# GridSearchCV(model, param_grid={'n_estimators':np.arange(5, 10, 1),
#                                 'max_depth': [3, 5, 7, 9]},
#                                 cv=StratifiedKFold(n_splits=5, shuffle=True))
# model.fit(x_train, y_train, sample_weight=class_weight)
#
# # 4. 模型预测
# y_pre = model.best_estimator_.predict(x_test)
#
# # 5. 模型评估
# print(classification_report(y_test, y_pre))


# --------------------------------------------
# # 1. 导入依赖包
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.utils.class_weight import compute_sample_weight
# from sklearn.preprocessing import LabelEncoder
#
# # 2. 数据读取及预处理
# data = pd.read_csv('./winequality-red.csv')
#
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
#
# # 将标签编码为 0..K-1（关键修复点）
# le = LabelEncoder()
# y_enc = le.fit_transform(y)   # 例如原本 3..8 会变成 0..5
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_enc, test_size=0.2, stratify=y_enc, random_state=22
# )
#
# # 如需保存中间数据，记得 index=False，避免产生额外列
# pd.concat([X_train, pd.Series(y_train, name='target')], axis=1).to_csv('./红酒品质分类_train.csv', index=False)
# pd.concat([X_test,  pd.Series(y_test,  name='target')], axis=1).to_csv('./红酒品质分类_test.csv', index=False)
#
# # 如果你是从上面两个CSV再读回：
# train_data = pd.read_csv('./红酒品质分类_train.csv')
# test_data  = pd.read_csv('./红酒品质分类_test.csv')
# X_train = train_data.drop(columns=['target'])
# y_train = train_data['target'].to_numpy()
# X_test  = test_data.drop(columns=['target'])
# y_test  = test_data['target'].to_numpy()
#
# # 样本权重（避免覆盖模块名）
# sample_w = compute_sample_weight(class_weight='balanced', y=y_train)
#
# # 3. 模型 + 网格搜索
# base_model = XGBClassifier(
#     objective='multi:softmax',          # 多分类直接输出类别
#     num_class=len(le.classes_),         # 类别数（稳妥起见显式指定）
#     random_state=22,
#     n_estimators=100,                   # 初值，网格里会覆盖
#     n_jobs=-1
# )
#
# param_grid = {
#     'n_estimators': np.arange(50, 201, 50),
#     'max_depth': [3, 5, 7, 9]
# }
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
# gs = GridSearchCV(
#     base_model,
#     param_grid=param_grid,
#     cv=cv,
#     scoring='f1_weighted',
#     n_jobs=-1
# )
#
# gs.fit(X_train, y_train, sample_weight=sample_w)
#
# # 4. 预测（用最佳模型）
# y_pred_enc = gs.best_estimator_.predict(X_test)
#
# # 如果希望按原始质量分数（3..8）来出报告：
# from sklearn.metrics import classification_report
# y_pred = le.inverse_transform(y_pred_enc)
# y_test_orig = le.inverse_transform(y_test)
# print("Best params:", gs.best_params_)
# print(classification_report(y_test_orig, y_pred))



import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# 1) 读取原数据
df = pd.read_csv('./winequality-red.csv')

# 保证目标列名正确
target_col = 'quality'
assert target_col in df.columns, "找不到质量列 'quality'"

# 2) 丢弃目标为缺失的行（若有）
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y_raw = df[target_col]

# 3) 标签编码成 0..K-1（XGBoost 多分类要求）
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 4) 划分数据（用编码后的 y 做 stratify）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=22
)

# 5) 样本权重（注意传入的是已编码的 y）
sample_w = compute_sample_weight(class_weight='balanced', y=y_train)

# 6) 模型与网格搜索
base = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    random_state=22,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)

gs.fit(X_train, y_train, sample_weight=sample_w)

# 7) 预测与还原标签
y_pred_enc = gs.best_estimator_.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)
y_test_orig = le.inverse_transform(y_test)

print("Best params:", gs.best_params_)
print(classification_report(y_test_orig, y_pred))
