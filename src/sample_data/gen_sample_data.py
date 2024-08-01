from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import json

# 데이터 생성
X, y = make_regression(n_samples=50, n_features=1, noise=0.1)

# 정수로 변환
X = (X * 7).astype(int)
y = (y * 7).astype(int)

# JSON 데이터 생성
data = [{"Feature": int(X[i][0]), "Target": int(y[i])} for i in range(len(X))]

# JSON 파일로 저장
with open('data.json', 'w') as f:
    json.dump(data, f)

plt.scatter(X, y, color='blue')
plt.title('Generated Data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()
