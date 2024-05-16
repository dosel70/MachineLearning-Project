<div align="center">
<h2>[2024] 머신러닝 분류 & 회귀 프로젝트 📊</h2>
pytorch, sklearn, 다양한 회귀 & 분류 모델들 , 그리고 차원축소, Lasso, Ridge ,Logistic Regression 등 다양한 기법을 사용하여, 프로젝트를 진행하였습니다.
</div>

## 목차
##  1.📌 [미국 주택 가격 예측 회귀분석 프로젝트(Click!)](https://github.com/dosel70/MachineLearning-Project/wiki/ML-Project-%E2%80%90-USA-House-Price-Predict-(Regression%E2%80%90LinearData))
  - #### 😊 프로젝트 설명 (미국 주택 가격 예측 프로젝트)
  - 해당 프로젝트는 선형 데이터로 이루어진 데이터셋으로 이루어져 있습니다.
    
  - 위 데이터세트에서는 주택이 속해있는 지역들의 수익과, 지역의 인구, 주택의 방 개수와 침실 개수 , 주택의 연식연도와 같은 Feature(독립변수)들을 활용하여 Price로 되어 있는 가격을 타겟데이터로 선정하여 회귀분석을 통해 예측하였습니다.
    
  - 위 미국 주택 가격 예측 데이터셋은 회귀분석을 통해 분석을 하였으며, 분석 결과 과적합은 발생하지 않았고, 다중공선성에도 큰 문제는 발생하지 않았습니다.
    
  - 회귀모델의 경우 LinearRegression의 성능점수(R2 Score)가 약 0.91로 매우 적합한 결과를 보여주었습니다.
    
  - 해당 프로젝트에서는 Ridge, Lasso를 활용하여 정규화 작업도 시도 하였으며, Pytorch로 각 Feature들의 가중치와 편향을 구하였고, 검증데이터셋과 훈련데이터셋의 MSE값을 비교하여 과적합 여부도 분석해보았습니다.
  
##  2. 📌 [제품 품질 점수 회귀 분석 프로젝트(Click!)](https://github.com/dosel70/MachineLearning-Project/wiki/ML-Project-%E2%80%90-Manufact-Quality-Rating-Predict-(Regression%E2%80%90Non-LinearData))
  - #### 😊 프로젝트 설명 (제품의 품질 평가 점수 예측 프로젝트)
  - 해당 프로젝트는 비선형 데이터로 이루어진 데이터셋으로 이루어져 있습니다.

  - 위 데이터세트에서는 제품 제조에 필요한 온도와, 압력, 그리고 온도와 압력을 곱한 Feature와 재료제조에 필요한 융합지표와 변환 지표와 같은 Feature(독립변수)들을 활용하여 Quality Rating(품질 평가 점수)를 타겟 데이터(종속변수)로 선정하여 회귀분석을 통해 예측하였습니다.
  
  - 위 제품 품질 평가 점수 예측 데이터셋은 회귀분석을 통해 분석을 하였으며, 몇개의 Feature가 매우 높은 다중공선성을 가지고 있는 것을 확인 하였고, 이러한 Feature들을 제거한 경우가 더 회귀모델들의 성능이 높게 나타났습니다. 그리고 교차검증에서 하이퍼파라미터튜닝을 하여서 과적합을 해소할 수 있었습니다.

  - 회귀모델의 경우 RandomForest Regressor와 Gradient Boosting Regressor의 성능점수(R2 Score)가 약 0.9994로 과적합을 해소 한 뒤의 성능 역시 매우 높게 나타났습니다.

  - 해당 프로젝트 에서는 Gridsearch CV와 KFold 교차검증을 수행하였고, 여기서 하이퍼파라미터 튜닝을 통해 검증데이터와 훈련데이터와의 손실값 차이를 줄여주어서 과적합을 해소하였습니다.  
  - 또, VIF로 다중공선성 점수를 산출하여 다중공선성 문제를 해소할 수 있었습니다.

  - 종속변수 데이터의 분포의 일반화를 위해 PowerTransformer, Log 치환 등을 통해 작업을 진행하였습니다.
  
##  3. 📌 [심부전 환자 분류 분석 (Logistic Regression) Click!](https://github.com/dosel70/MachineLearning-Project/wiki/ML-Project-%E2%80%90-HeartFailure-Classifier-Project)   
  - #### 😊 프로젝트 설명 (심부전 질환 환자 예측 분류 프로젝트)
  - 해당 프로젝트는 로지스틱 회귀를 이용하여 분류 분석을 하였습니다.

  - 위 데이터세트는 환자들의 심박수, 폐혈증 유무, 콜레스테롤 수치와 같은 Feature(독립변수)들을 활용하여 HeartDisease(종속변수)를 타겟데이터로 선정하여 회귀분석을 통해 예측하였습니다.

  - 위 심부전 환자 분류 예측 프로젝트는 로지스틱회귀분석을 통해 작업을 하였으며, 몇개의 Feature가 기존 타겟데이터에 대해 상관관계가 매우 낮았고, 로지스틱회귀훈련결과에서도 Feature들의 중요도가 낮은 Feature들이 있었습니다. 이러한 Feature들을 제거한 경우가 더 회귀모델들의 성능이 높게 나타났습니다. 그리고 교차검증에서 하이퍼파라미터튜닝을 하여서 과적합을 해소할 수 있었습니다.

  - Logistic Regression (로지스틱회귀분석)의 경우 정확도: 0.8424,  F1: 0.8585 가 나왔으며, 대체적으로 좋은 성능으로 나타났습니다.

  - 해당 프로젝트에서는 lda로 차원축소를 수행하였고, 따로 pytorch로 검증데이터와 훈련데이터와의 손실값 차이를 비교하여, 과적합을 해소할 수 있었습니다.
  
##  4. 📌 [관상동맥질환 환자 분류 분석 (High Dimension - Dimension Reduction) Click!](https://github.com/dosel70/MachineLearning-Project/wiki/ML-Project-%E2%80%90-Coronary-Artery-Predict-(Dimension_reduction))   
  - #### 😊 프로젝트 설명 (관상동맥질환 환자 예측 분류 프로젝트)
  - 해당 프로젝트는 고차원데이터에서 pca , lda 차원축소를 활용하여 분류 분석을 하였습니다.

  - 위 데이터세트는 환자들의 질병들과  같은 Feature들을 활용하여 Cath(종속변수 = 관상동맥질환 이진분류)를 타겟 데이터로 선정하여 분류분석을 통해 예측하였습니다.

  - 해당 데이터세트에서는 pca로 차원축소를 했을 때 보다 lda로 차원축소를 하고 나서 가장 성능이 좋았던 분류모델인 Randomforest로 작업을 하였을 때, 더 성능이 좋았으므로,
  - lda 차원축소를 통해서 작업을 진행하였습니다.

  - 위 고차원 분류 프로젝트에서는 로지스틱회귀분석과 RandomForest 분류 모델로 작업을 하였으며, 로지스틱회귀분류 보다 randomforest 분류모델로 작업하였을 때, 더 성능이 좋았습니다.
---

