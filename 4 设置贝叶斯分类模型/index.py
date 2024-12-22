from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# 加载数据集并划分训练集和测试集
def load_and_split_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分训练集和测试集，测试集占比30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

#-------------------------------------Begin------------------------------------------------
# 创建并训练朴素贝叶斯分类器
def train_classifier(X_train, y_train):
    # 提示：创建高斯朴素贝叶斯分类器对象
    #classifier =
    classifier = GaussianNB()
    # 提示：使用训练集数据训练分类器
    #classifier.fit(...)
    classifier.fit(X_train, y_train)
    return classifier
#--------------------------------------End-----------------------------------------------

if __name__ == "__main__":
    # 加载并划分数据集
    X_train, X_test, y_train, y_test = load_and_split_dataset()

    # 训练分类器
    classifier = train_classifier(X_train, y_train)
    classifier_type = classifier.__class__.__name__
    print(classifier_type)