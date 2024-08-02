import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성
# 1차원 feature와 target을 갖는 샘플 데이터 생성
np.random.seed(0)  # 재현성을 위해 시드 설정
X = 2 * np.random.rand(1000, 1)  # 0에서 2 사이의 난수로 1000개의 샘플 생성
y = 4 + 3 * X + np.random.randn(1000, 1)  # 실제 관계: y = 4 + 3x + 잡음

# 2. 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 3. 계수와 절편 추출
coefficient = model.coef_
intercept = model.intercept_

# 결과 출력
print("Coefficient (기울기):", coefficient)
print("Intercept (절편):", intercept)

