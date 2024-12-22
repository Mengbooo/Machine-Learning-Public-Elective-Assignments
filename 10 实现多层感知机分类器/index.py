import numpy as np
# from step2.model import MLP, relu, softmax, cross_entropy_loss
# ! 注意：这里导入的step2里面的model.py为touge平台给出
# 训练时间约30秒       
class MyMLP(MLP):
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化多层感知机（MLP）模型, 构建网络架构
        :param input_size: 输入层向量长度（忽略图片二维信息，被拉直后的向量长度）
        :param hidden_size: 隐藏层输入长度
        :param output_size: 输出层向量长度
        :return: 无
        """
        self.input_size = input_size
        # a1是隐藏层的输出值，a2是输出层的输出值
        self.a1 = None
        self.a2 = None
        # 初始化输入层到隐藏层的权重矩阵w1和偏置向量b1
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        # 初始化隐藏层到输出层的权重矩阵w2和偏置向量b2
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """
        前向传播计算
        :param hidden_size: 输入数据（忽略灰度图片二维信息，已拉直图片）
        :return: 输出层的激活值，经过 softmax 转换为概率分布
        """
        # ********* Begin ********* #
        # 根据提示，计算隐藏层的线性组合，并应用ReLU激活函数赋值给self.a1
        z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(z1)
        # 根据提示，计算输出层的线性组合，并应用softmax激活函数赋值给self.a2
        z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(z2)
        # ********* End ********* #
        return self.a2

    def backward(self, X, y, learning_rate):
        """
        反向传播更新权重和偏置
        :param X: 输入数据
        :param y: 真实标签
        :param learning_rate: 学习率，用于更新参数的步长
        :return: 无
        """
        super().backward(X, y, learning_rate)

    def fit(self, X, y, learning_rate, num_epochs):
        """
        模型训练过程
        :param X: 输入数据，形状为 (样本数, 输入层节点数)
        :param y: 真实标签，形状为 (样本数, 类别数)
        :param learning_rate: 学习率，用于更新参数的步长
        :param num_epochs: 训练的轮数
        :return: 无
        """
        for epoch in range(num_epochs):
            # ********* Begin ********* #
            # 执行前向传播, 计算结果赋值给y_pred
            y_pred = self.forward(X)
            # ********* End ********* #

            # 计算交叉熵损失
            loss = cross_entropy_loss(y_pred, y)
            
            # 执行反向传播并更新参数
            self.backward(X, y, learning_rate)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')
            
            