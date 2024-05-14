<div align="center">
<h2>[2024] 머신러닝 분류 & 회귀 프로젝트 📊</h2>
pytorch, sklearn, 다양한 회귀 & 분류 모델들 , 그리고 차원축소, Lasso, Ridge ,Logistic Regression 등 다양한 기법을 사용하여, 프로젝트를 진행하였습니다.
</div>

## 목차
  - [미국 주택 가격 예측 회귀분석 프로젝트](#미국-주택-가격-예측-회귀분석)
  
  - [제품 품질 점수 회귀 분석 프로젝트](#제품-품질-점수-회귀-분석)  
  
  - [심부전 환자 분류 분석 (Logistic Regression)](#심부전-환자-분류-분석 (Logistic Regression))   
  
  - [관상동맥질환 환자 분류 분석 (High Dimension - Dimension Reduction)](#관상동맥질환-환자-분류-분석 (High Dimension - Dimension Reduction))   
  
## 미국 주택 가격 예측 회귀분석
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
  - [pytorch 및 sklearn 라이브러리 회귀 모델들로 분석](#✨-1-Cycle)
  - [과적합 유무 분석](#과적합-분석)
  - [각 모델들의 성능(MSE, RMSE, R2 Score) 분석](#💡-Total-Result)

## 전처리 작업
- ✏️ 해당 데이터세트 에서는 결측치 및 중복치가 없었으며 컬럼들의 분포 또한 아래와 같이 정규분포화가 잘 되어 있습니다.
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/99b355c9-092b-409e-934d-56d0ce73e50e' width="600px">

## correlation 종속변수와의 상관관계 분석
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/b675d383-f5a2-4de2-9435-91f38c033a99' width="600px">  

위 이미지와 같이 Area Income 즉 지역의 수입이 가장 상관관계가 높았으며, 가장 상관관계가 낮은 Feature는 침실의 개수였습니다.   
후에 OLS(최소제곱법)와 VIF 점수를 토대로 correlation과 비교하여 분석하겠습니다.

### 📌 전처리 완료
## ✨ 1 Cycle  
> ### sklearn(사이킷런 라이브러리) 으로 LinearRegression 및 이외 회귀 모델 훈련
- #### LinearRegression 훈련 Sourcode PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/e18c9729-c907-4ebb-a002-f026b811edbf' width="800px" style="margin-bottom:10px">  

- #### PolynomialRegression 훈련 (degree->2) PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/d6cb6db9-c0be-4bad-8766-ee2483835532' width="800px" style="margin-bottom:10px">

- #### RandomForest, GradientBoosting, XGBM, LGBM Regressor PNG
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/39fe4ebc-1efb-43ec-8d78-4fbae263b9f1' width="800px" style="margin-bottom:10px">

- ### 💡 Polynomial Regression Distribution (다항회귀모델에서 최적의 항 차수를 찾기)
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/8b0b7ae4-3757-4ad3-8296-08e807045905' width="1000px" style="margin-bottom: 10px">

- ### 💡 Total Regressor's R2 Score (전체 회귀모델 성능 분석 (다항회귀 제외))
  <img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/ebf6d888-678e-41f6-a146-6f32b7e1b005' width="1000px" style="margin-bottom:10px">
