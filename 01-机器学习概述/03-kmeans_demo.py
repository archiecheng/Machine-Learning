"""
## 无监督聚类API - Kmeans
#   1   导包 sklearn.cluster.KMeans sklearn.datasets.make_blobs
    2   创建数据集
    2-1 展示数据集
    3   实例化Kmeans模型并预测
    3-1 展示聚类效果
    4   评估3种聚类效果好坏
"""

# 1. 导入依赖包
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
def dm03_kmeans_demo():
    # 2. 创建数据
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=11)
    plt.figure()
    plt.scatter(x[:,0], x[:,1], marker='o')
    plt.show()
    # 3. 模型实例化
    kmeans_cls = KMeans(n_clusters=2, init='k-means++', n_init='auto')
    # 4. 模型预测
    y_pred = kmeans_cls.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c = y_pred)
    plt.show()
    # 5. 模型评估
    print(calinski_harabasz_score(x, y_pred))



    # 3. 模型实例化
    kmeans_cls = KMeans(n_clusters=3, init='k-means++', n_init='auto')
    # 4. 模型预测
    y_pred = kmeans_cls.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c = y_pred)
    plt.show()
    # 5. 模型评估
    print(calinski_harabasz_score(x, y_pred))


    # 3. 模型实例化
    kmeans_cls = KMeans(n_clusters=4, init='k-means++', n_init='auto')
    # 4. 模型预测
    y_pred = kmeans_cls.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c = y_pred)
    plt.show()
    # 5. 模型评估
    print(calinski_harabasz_score(x, y_pred))




if __name__ == '__main__':
    dm03_kmeans_demo()