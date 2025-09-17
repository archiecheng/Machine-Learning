# 1. 导入依赖包

import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 2. 读取数据及数据预处理
# 2.1 读取数据
data = pd.read_csv('./书籍评价.csv', encoding='utf-8')
print(data)

# 2.2 添加 labels 列
data['label'] = np.where(data['评价'] == '好评', 1, 0)
# print(data)
y = data['label']

# 2.3 设置停用词
stop_words = []
with open('stopwords.txt', encoding='utf-8') as file:
    lines = file.readlines()
    stop_words = [line.strip() for line in lines]

stop_words = list(set(stop_words))

# 2.4 分词
word_list = [','.join(jieba.lcut(line)) for line in data['内容']]

# 2.5 词频统计
transform = CountVectorizer(stop_words=stop_words)
x = transform.fit_transform(word_list)
names = transform.get_feature_names_out()

x = x.toarray()

# 2.6 数据集划分
x_train = x[:10, :]
y_train = y.values[0:10]

x_test = x[10:,:]
y_test = y.values[10:]


# 3. 模型训练
model = MultinomialNB(alpha=1)
model.fit(x_train,y_train)

# 4. 模型预测
y_predict = model.predict(x_test)
print(y_predict)

# 5. 模型评估
print(model.score(x_test, y_test))


