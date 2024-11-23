from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def classification(X, y, k):
    """
    KNN分类流程
    :param X: 数据集特征
    :param y: 数据集标签
    :param k: 邻居数量
    :return: 当前邻居节点的评分结果
    """
    # ********* Begin ********* #
    # 令random_state=2024，划分出40%的数据作为测试数据，剩下作为训练数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2024)
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 实例化KNN分类器，设置n_neighbors为k
    knn = KNeighborsClassifier(n_neighbors=k)
    # 根据训练集训练模型
    knn.fit(X_train_scaled, y_train)
    # 预测测试集
    y_pred = knn.predict(X_test_scaled)
    return y_pred
    # ********* End ********* #
