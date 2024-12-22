import numpy as np

def relu(x):
    """
    实现ReLU激活函数
    :param x: 输入，一维向量，如[-2, -1, 0, 1, 2, 3]
    :return: 返回一个与输入数组相同形状的数组，包含ReLU函数的输出
    """
    # ********* Begin ********* #
    # 根据数学公式实现ReLU函数
    return np.maximum(0, x)
    # ********* End ********* #

def softmax(x):
    """
    实现softmax函数，用于将输入转换为概率分布
    :param x: 输入，一维向量，如[-2, -1, 0, 1, 2, 3]
    :return: 返回一个与输入数组相同形状的数组，包含softmax函数的输出
    """
    # ********* Begin ********* #
    # 根据数学公式实现softmax函数
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
    # ********* End ********* #


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss