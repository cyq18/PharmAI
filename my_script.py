from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 示例数据：年龄和是否购买
X = np.array([[22], [25], [47], [54], [46]])  # 年龄
y = np.array([0, 1, 1, 0, 1])  # 是否购买

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

print(f"预测值: {y_pred}")
