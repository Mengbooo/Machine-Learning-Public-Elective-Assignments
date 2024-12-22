from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 提示：定义一个函数加载数据集并划分训练集和测试集
def load_and_split_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 提示：使用train_test_split函数划分训练集和测试集，测试集占比30%，random_state=42.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# 提示：定义一个函数创建并训练朴素贝叶斯分类器
def train_classifier(X_train, y_train):
    # 创建高斯朴素贝叶斯分类器对象
    classifier = GaussianNB()

    # 使用训练集数据训练分类器
    classifier.fit(X_train, y_train)

    return classifier
#-------------------------------------Begin------------------------------------------------
# 提示：定义一个函数使用训练好的分类器进行预测并评估准确率
def predict_and_evaluate(classifier, X_test, y_test):
    # 提示：对测试集进行预测
    #y_pred = classifier.predict(...)
    y_pred = classifier.predict(X_test)

    # 提示：计算准确率
    #accuracy = accuracy_score(...)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
#--------------------------------------End-----------------------------------------------


if __name__ == "__main__":
    # 提示：加载并划分数据集
    X_train, X_test, y_train, y_test = load_and_split_dataset()

    # 提示：训练分类器
    classifier = train_classifier(X_train, y_train)

    # 提示：预测并评估准确率
    accuracy = predict_and_evaluate(classifier, X_test, y_test)

    print("Accuracy:", accuracy)