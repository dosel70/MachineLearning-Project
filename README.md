<div align="center">
<h2>[2024] 머신러닝 분류 & 회귀 프로젝트 📊</h2>
pytorch, sklearn, 다양한 회귀 & 분류 모델들 , 그리고 차원축소, Lasso, Ridge ,Logistic Regression 등 다양한 기법을 사용하여, 프로젝트를 진행하였습니다.
</div>

## 목차
  - [미국 주택 가격 예측 회귀분석 프로젝트](https://github.com/dosel70/MachineLearning-Project/wiki/ML-Project-%E2%80%90-USA-House-Price-Predict-(Regression%E2%80%90LinearData))
  
  - [제품 품질 점수 회귀 분석 프로젝트](#제품-품질-점수-회귀-분석)  
  
  - [심부전 환자 분류 분석 (Logistic Regression)](#심부전-환자-분류-분석 (Logistic Regression))   
  
  - [관상동맥질환 환자 분류 분석 (High Dimension - Dimension Reduction)](#관상동맥질환-환자-분류-분석 (High Dimension - Dimension Reduction))   
---

# 📈 선형 회귀 분석 프로젝트  

## 미국 주택 가격 예측 회귀분석
<img src="https://blog.kakaocdn.net/dn/r7ewt/btq8TSge90l/Ie7lNsaBoAoXWafOnsbI60/img.jpg">    

  ### 📌 데이터 세트 주제 
  - 해당 지역의 수익, 방의 개수, 침실의 개수 등의 Feature들을 활용하여 주택 가격의 값을 회귀분석하여 예측합니다.
  #### 📌 컬럼별 설명
  - Avg. Area Income : 해당 지역의 평균 수익
  - Avg. Area House Age : 해당 지역 내 각 주택의 평균 연식을 나타내는 특성을 나타내는 Feature
  - Avg. Area Number of Rooms : 주택의 평균 방 개수 
  - Avg. Area Number of Bedrooms : 주택의 평균 침실 개수 
  - Area Population : 지역의 인구
  - **Price : 주택 가격 (**Target 데이터(독립변수)**)**
  - Address : 주택 주소
    
  ### ✏️ 미국 주택 가격 예측 회귀 프로젝트 진행 방향성
  - [데이터 전처리 (결측치, 중복된 데이터, 이상치 등 제거 및 일반화 작업)](#전처리-작업)
  - [독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)
  - [pytorch 및 sklearn 라이브러리 회귀 모델들로 분석](#📌-전처리-완료)
  - [1 Cycle - Pytorch & 사이킷런 라이브러리로 회귀모델 데이터 회귀 분석](#1-Cycle)
  - [2 Cycle - Pytorch로 과적합 분석 ](#2-Cycle)
  - [3 Cycle -Sklearn으로 과적합 분석 ](#3-Cycle)
  - [4 Cycle - OLS 회귀분석 & 다중공선성 확인](#4-Cycle)
  - [5 Cycle - 다중공선성 해소한 데이터세트로 과적합 여부 확인](#5-Cycle)
  - [6 Cycle - Lasso & Ridge (Regularized_linear_regression 정규화된 선형 회귀) 분석](#6-Cycle)
  - [7 Cycle - 회귀모델 데이터 회귀 분석 및 기존 데이터세트와 성능 비교](#7-Cycle)
  - [최종 결론](#Total-Result)

## 데이터세트(csv파일 PNG) <USA House Price Predict>
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/2d1b047e-6a98-450b-afa3-5743575044b2' width="600px">

## 전처리 작업
- ✏️ 해당 데이터세트 에서는 결측치 및 중복치가 없었으며 컬럼들의 분포 또한 아래와 같이 정규분포화가 잘 되어 있습니다.
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/99b355c9-092b-409e-934d-56d0ce73e50e' width="600px">

## correlation 종속변수와의 상관관계 분석
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/b675d383-f5a2-4de2-9435-91f38c033a99' width="600px">  

위 이미지와 같이 Area Income 즉 지역의 수입이 가장 상관관계가 높았으며, 가장 상관관계가 낮은 Feature는 침실의 개수였습니다.   
후에 OLS(최소제곱법)와 VIF 점수를 토대로 correlation과 비교하여 분석하겠습니다.  
[4 Cycle 로 이동](#4-Cycle)

### 📌 전처리 완료  

## 1 Cycle  
> ### sklearn(사이킷런 라이브러리) & Pytorch 로 LinearRegression 및 이외 다변량 회귀 모델 훈련

- #### Pytorch로 다중선형회귀 분석을 통해 각 Feature의 가중치(기울기)와 편향 구하기
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/ad0c4d3c-297b-4072-8143-4e9c5654faa1' width="800px" style="margin-bottom:10px">  
  
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/1aabf95b-ba49-4a04-8452-c83f71479f91' width="800px" style="margin-bottom:10px">   

  > 위 이미지와 같이 다중선형회귀에서 각 Feature들의 가중치와 편향을 산출 하였습니다.

    
- #### LinearRegression 훈련 Sourcode PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/e18c9729-c907-4ebb-a002-f026b811edbf' width="800px" style="margin-bottom:10px">  
  
  > 위 이미지와 같이 coef_를 통해서 각 Feature 마다 기울기와 해당 데이터의 편향을 산출하였습니다.  
  
- #### PolynomialRegression 훈련 (degree->2) PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/d6cb6db9-c0be-4bad-8766-ee2483835532' width="800px" style="margin-bottom:10px">

- #### RandomForest, GradientBoosting, XGBM, LGBM Regressor PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/39fe4ebc-1efb-43ec-8d78-4fbae263b9f1' width="800px" style="margin-bottom:10px">

- #### 💡 Polynomial Regression Distribution (다항회귀모델에서 최적의 항 차수를 찾기)
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/8b0b7ae4-3757-4ad3-8296-08e807045905' width="1000px" style="margin-bottom: 10px">  
  > 다항회귀로 분석을 할때에는 해당 데이터에서는 항 차수가 2차항에서 3차항으로 설정할때 가장 성능이 높게 나왔습니다.

- #### 💡 Total Regressor's R2 Score (전체 회귀모델 성능 분석 (다항회귀 제외))
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/ebf6d888-678e-41f6-a146-6f32b7e1b005' width="1000px" style="margin-bottom:10px">  


### 📃 1 Cycle Result
> Linear Regression 선형 회귀 모델의 경우 R2 Score가 0.9211로 모든 회귀모델 중에서 가장 높은 수치를 기록하였습니다.
> 다른 회귀모델들도 모두 0.8 이상의 성능 점수를 가졌기 때문에 모두 성능이 좋다고 볼 수 있지만, 선형회귀모델로 분석하였을 때 성능이 가장 높게 나오므로, 해당 데이터는 선형 데이터에 가깝습니다.

## 2 Cycle
> ### Pytorch로 손실값 산출 및 과적합 분석
> 훈련데이터와 검증데이터간의 손실값 차이를 시각화하여 과적합의 여부를 확인 할 수 있습니다.
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/eb63cdf8-fd5c-460b-b74c-39a49d695fd6" width="600px">
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/506112a8-72ce-4f26-8c04-dc35c10bec53" width="600px">
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/83953fe2-3e50-4871-b9f5-c62c2c2780b6" width="600px">

#### ✨ 훈련 데이터와 검증 데이터의 손실값 차이 시각화 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/d233c887-2fa2-4d1c-901a-264d4cdf8f4d" width="700px">   

### 📃 2 Cycle Result
> #### **위 시각화 이미지와 같이 훈련데이터 loss값과 검증데이터의 loss값이 epoch 횟수가 증가함에 따라 거의 동일한 loss값으로 나오는 것을 볼 수 있습니다.**  
> #### 📌 따라서, 본 데이터셋은 OLS 회귀 분석이나 다중공선성 검토를 수행하기 전부터 과적합 문제가 존재하지 않는 것으로 보입니다.   
> #### 이는 데이터의 특성상 독립변수들이 종속변수에 대해 유의미한 영향을 미치며, 다중공선성 또한 낮아 모델의 안정성을 보장하고 있음을 시사합니다.

## 3 Cycle
> ### Sklearn으로 cross_val_score 라이브러리 함수를 활용하여 과적합 분석
>  `from sklearn.model_selection import cross_val_score`

> cross_val_score 함수를 활용하여 훈련데이터와 검증데이터 간의 손실값 차이를 시각화하여 과적합의 여부를 확인 할 수 있습니다.
<div>
  <img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/7497fd98-499b-4634-be7e-b39a9f6554df" width="600px" style="margin-bottom: 30px">
</div>

  
> **시각화 이미지 (훈련데이터(orange) vs 검증데이터(blue))**
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/31891532-4c66-44bf-a82a-a9ca4b690a85" width="600px" style="margin-bottom: 10px">
>

### 📃 3 Cycle Result
> #### Sklearn에서 cross_val_score 이라는 라이브러리 함수를 사용하여 과적합을 분석하였을 때도 마찬가지로 과적합이 존재하지 않는 다는 것을 알 수 있었습니다.

## 4 Cycle
> ### OLS(최소제곱법) & VIF(다중공선성 점수) 산출 & Feature 제거 및 보류 작업
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/d1d7e228-ae9a-4915-9705-d7029e459dfe" width="600px">   

- OLS 분석 결과 위에서 correlation에서 가장 타겟데이터와의 상관관계가 낮았던 침실의 개수의 P-Value 값이 다른 Feature들에 비해 약간 높은 수치를 보였습니다. [독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)
  
- 이는 침실의 개수가 주택 가격에 대해 다른 독립변수들에 비해 영향력이 적다는 것을 의미합니다.
- 다음으로 VIF 점수를 통해서 독립변수들의 다중공선성을 확인 해 보겠습니다.
  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/21dcaff2-e793-4181-9c6e-2a75e0887836" width="600px">  

> 위 이미지와 같이 주택의 방의 개수의 다중공선성 점수가 가장 높은 것을 확인 할 수 있습니다.
> 방의 개수의 경우 침실의 개수를 나타내는 Feature와 연관성이 높기 때문에 아마 침실의 개수 Feature 때문에 다중공선성 점수가 높아진 것으로 판단 할 수 있습니다.
> 결과적으로 종속변수와의 상관관계가 더 낮은 침실의 개수를 제거 하면 방의 개수의 다중공선성을 해소할 수 있을 것입니다.

 <img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/c748e4f4-0930-40bb-aa68-6a71fa06fd56" width="500px">  

 ### 📌 Feature 제거 후 OLS 분석 결과
 <img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/f160bf16-017d-41d9-9992-049a76ed57cc" width="500px">  
 
 > OLS 회귀분석 결과 R2 Score는 기존 0.964로 동일 한 것을 볼 수 있습니다.  

 ### 📌 Feature 제거 후 VIF 점수 산출  
 <img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/1c896e34-ac19-400e-989c-633dea4ff320" width="500px">  
 
 > VIF 산출 결과 주택의 방 개수 Feature의 다중공선성 점수가 많이 감소하여 이전에 비해 다중공선성 문제를 해소할 수 있었습니다.

 ### 📃 4 Cycle Result
 > OLS 회귀분석과 VIF점수를 통한 다중공선성 문제를 해소를 하였습니다.
>
> 위 작업을 통해 회귀모델의 성능을 평가하고, 과적합 여부를 확인 하겠습니다.

## 5 Cycle  
> ### Feature(주택의 침실의 개수) 제거 후 과적합 확인

#### Pytorch로 과적합 분석 시각화 (Train Data loss vs Validation Data loss)
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/c517060f-cceb-4432-97e0-a6c8353aa703" width="700px">  

> Pytorch로 과적합 분석을 시각화 한 결과 기존 Feature 제거 전 데이터와 마찬가지로 과적합은 존재하지 않았습니다.  

#### Sklearn으로 과적합 분석 시각화 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/fdaea798-2e20-425e-9d81-43abdeed780a" width="700px">  

> cross_val_score 함수를 활용하여 과적합을 분석한 결과 역시 Feature를 제거해도 과적합은 발생하지 않았습니다.

### 📃 5 Cycle Result  
> 해당 데이터세트에서 다중 공선성을 해소하기 위해 침실의 개수를 나타내는 Feature를 제거하여도, 과적합 문제는 발생하지 않습니다.
> 다음으로는 Feature 제거 전 데이터에서 Lasso, Ridge 회귀를 사용하여 모델의 규제 효과와 성능을 분석하겠습니다.

## 6 Cycle
> ### Feature 제거 전 원래 데이터셋에서 Lasso, Ridge 회귀를 활용하여 일반화 성능 평가

- #### Lasso
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/d5014bfd-41d2-4ea6-8038-062799c41c80" width="600px">

> #### 📌 다중공선성으로 보이는 Feature 제거하기 전 데이터에서 Lasso 회귀분석을 한 결과 다양한 규제 값에서도 lasso 회귀 모델의 성능지표가 거의 일정하게 유지되었습니다.
> #### 이는 해당 데이터셋이 과적합 문제를 겪지 않고 있으며, 규제 파라미터를 변경해도 모델 성능에 큰 영향을 미치지 않음을 알 수 있습니다.
> #### **따라서 해당 데이터는 이미 최적화되어있기에, 추가적인 규제는 필요하지 않을 수 있습니다.**
> ##### 추가적으로 이러한 결론을 바탕으로 다음 단계로 Ridge 회귀를 적용하여 모델의 성능을 비교해보겠습니다.

- #### Ridge
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/2c147fec-9722-4747-832c-03c98b09d23c" width="600px">  

> #### 📌 Ridge 회귀 분석도 마찬가지로 다양한 규제 파라미터 값을 주어도 회귀모델의 성능지표가 거의 일정하게 유지되었으며, 특히 규제값이 1000일 때, 성능이 꽤 저하됨을 관찰했습니다. 이러한 결과는 1000과 같은 너무 큰 규제는 피해야 함을 의미합니다.
>
> #### 결론적으로 해당 데이터셋이 과적합 문제를 겪지 않고 있으며, Ridge 회귀를 통해서 따로 VIF를 산출해서 제거할 필요 없이 RIdge 회귀를 통해 다중공선성을 잘 해결할 수 있음을 알 수 있습니다.

### 📃 6 Cycle Result
> 해당 데이터셋은 과적합 문제가 없으며, Lasso와 Ridge 회귀 모델에서 다양한 규제(alpha) 값에 대해 성능이 안정적으로 유지되었습니다. 이는 데이터가 잘 정제되어 있고, 다중공선성이 크지 않음을 나타냅니다. 따라서, 추가적인 규제 조정 없이도 모델의 성능은 충분히 양호하며, 과도한 규제는 성능 저하를 초래할 수 있습니다.

## 7 Cycle 
> ### 다중공선성 Feature를 제거한 데이터와 기존 데이터와의  회귀모델 성능 비교

- 기존 데이터 회귀모델 성능 비교 그래프
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/46b63e48-1762-4b18-8ab7-2f5c5d1b7bd6" width="800px">

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/a0f9e3d0-a72c-4ba2-a774-a610d6162d4c" width="800px">  

## Total Result
- ### ✨ 최종결론
- 다중공선성을 해소하기 위해 Feature를 제거한 데이터셋과 기존 데이터셋과의 회귀모델들의 성능을 비교한 결과 큰 변화폭은 없었으며, Decisiton Tree Regressor, XGBM과 LGBM 회귀모델의 성능을 소폭 증가시킬 수 있었지만, RandomForest회귀모델과 GradientBoosting Regressor회귀모델의 성능은 소폭 감소하였습니다.
- 그러나 아주 큰 변화라고 볼 수 없기 때문에, 기존 데이터셋 자체의 데이터가 잘 정제되어 있음을 알 수 있었습니다.
