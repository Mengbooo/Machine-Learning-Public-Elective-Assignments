from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. 加载数据集（以乳腺癌数据集为例）
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# 2. 数据预处理
# 使用train_test_split划分训练集和测试集，设置test size=0.2，random state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ntr = X_train.shape[0]
nts = X_test.shape[0]
print("num samples train = %d, test = %d" % (ntr, nts))