from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. 加载数据集（以乳腺癌数据集为例）
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

#使用train_test_split来划分X和y，并设置test_size = 0.2, random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 kernel = RBF,random state = 42 (RBF核的SVM)
svm = SVC(kernel='rbf', random_state=42)

# 超参数调优

#设置param_grid的参数c和gamma，格式为‘name’: [value1, value2...] C的值为0.1, 1, 10, 100 gamma的值为1, 0.1, 0.01, 0.001
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01,0.001]
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
