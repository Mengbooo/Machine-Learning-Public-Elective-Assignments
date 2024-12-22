from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#-------------------------------------Begin------------------------------------------------
# 加载数据集并划分训练集和测试集
def load_and_split_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 提示：使用train_test_split函数划分训练集和测试集，测试集占比30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)# 填写代码
    return X_train, X_test, y_train, y_test
#--------------------------------------End-----------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_dataset()
    ntr = X_train.shape[0]
    nts = X_test.shape[0]
    print("num samples train = %d, test = %d" % (ntr, nts))