import json
import numpy as np
from sklearn.linear_model import LinearRegression

# data.json 파일에서 데이터를 읽어옴
with open('data.json', 'r') as file:
    data = json.load(file)

# 데이터를 numpy 배열로 변환
X = np.array([entry['Feature'] for entry in data]).reshape(-1, 1)
y = np.array([entry['Target'] for entry in data])

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X, y)

# '70'를 입력으로 받아 예측
input_value = np.array([[70]])
prediction = model.predict(input_value)

print(f"Input: {input_value[0][0]}, Predicted Target: {prediction[0]}")

