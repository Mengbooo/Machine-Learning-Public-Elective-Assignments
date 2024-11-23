任务描述

实现KNN分类器的执行流程，用于对潜在病人进行分类。
完成分类（classification）函数。

相关知识
KNN分类器
在上一关中，我们已经实现了一个KNN分类器，并通过了测试。同样，sklearn中也已经实现了具有相同的功能的KNN分类器，为了熟悉KNN分类器的使用流程，我们可以直接调用其接口对数据进行分类：

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 初始化KNN分类器，设定邻居数为3
knn = KNeighborsClassifier(n_neighbors=3)
# 训练KNN分类器
knn.fit(X_train, y_train)
# 预测测试集
y_pred = knn.predict(X_test)
标准化
如果一个数据集中有两个特征，身高（以米为单位）和收入（以万元为单位），未经标准化的情况下，收入的数值范围远大于身高，这会使得收入在距离计算中占主导地位，忽略了身高的重要性。由于不同特征的尺度会影响距离计算，从而导致模型结果偏差，需要对特征进行标准化。

Z Score标准化通过删除平均值和缩放到单位方差来标准化特征，并将标准化的结果的均值变成0，标准差为1。Z Score是最常用的标准化手段，sklearn也提供了相应的接口：

from sklearn.preprocessing import StandardScaler
# 创建一个示例数据集
data = [[1.70, 50],[1.80, 60],[1.75, 65],[1.60, 45]]
# 初始化StandardScaler
scaler = StandardScaler()
# 对data进行标准化
data_scaled = scaler.fit_transform(data)
# 使用对应标准化器对新数据处理
data_new = scaler.transform([[1.65, 55],[1.85, 70]])
# 打印标准化后的数据
print(data_scaled)
print(data_new)
打印如下：

[[-0.16666667,-0.63245553],[ 1.16666667,0.63245553],[ 0.5,1.26491106],[-1.5 ,-1.26491106]]
[[-0.83333333,0.0],[1.83333333,1.8973666]]
编程要求
根据提示，在右侧编辑器中完善代码，实现KNN分类器的执行流程，平台会对分类结果进行评估

测试说明
本关占期末总成绩的2.5分，其中编译通过不返回错误获得0.5分，classification函数测试分类精度≥0.70获得2分。若程序逻辑大致正确但编译未通过获得1分。