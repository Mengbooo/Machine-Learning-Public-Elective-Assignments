import numpy as np


def distance(x1, x2):
    """
    距离函数
    :param x1,x2: 向量
    :return: 距离值
    """
    # ********* Begin ********* #
    # 计算并返回x1与x2之间的距离

    # ********* End ********* #


class kNNClassifier(object):
    def __init__(self, k=5):
        """
        初始化函数
        :param k: kNN算法中的k, 默认为5
        """
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.x = None
        # 用来存放训练标签，类型为ndarray
        self.y = None

    def fit(self, x, y):
        """
        kNN算法的训练过程
        :param x: 训练集数据，类型为ndarray
        :param y: 训练集标签，类型为ndarray
        :return: 无返回
        """
        # ********* Begin ********* #

        # ********* End ********* #

    def predict(self, x):
        """
        kNN算法的预测过程
        :param x: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        """
        # ********* Begin ********* #
        # 计算测试样本与所有训练样本的距离

        # 按照距离排序，查找距离最近的k个训练样本

        # k个训练样本中出现次数最多的标签作为预测结果

        # 返回所有测试样本的预测标签

        # ********* End ********* #
