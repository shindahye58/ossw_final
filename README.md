# FINAL PROJECT

> **Brain Tumor Classification using by method in scikit learn package**

*my Evaluation environment*

>Numpy version: 1.26.2
>
>scikit learn version: 1.3.2
>
>scikit image version: 0.22.0

## Using model
> **Support Vector Machine, SVM** : 지도 학습(Supervised learning) 알고리즘 중 하나로 분류(Classification)와 회귀(Regression) 문제에 사용되는 강력하고 널리 쓰이는 알고리즘
>
> - 작동원리: 
## training model process

**Optimize hyper-parameters**

  - **model's hyper-parameters**

  - **grid search**

    >grid search란?  머신 러닝에서 모델의 최적 하이퍼파라미터를 찾기 위한 기술 중 하나,  이는 여러 가지 하이퍼파라미터 조합을 시도하여 모델의 성능을 평가하고, 최적의 조합을 찾는 과정
    >
    > 하이퍼파라미터 그리드 생성 🠒 모델 훈련 및 검증 🠒 최적 하이퍼 파라미터 선택 🠒 모델 평가
    >
```
# KNN 모델
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1, 2, 3, 5],# 이웃 수에 대한 후보 값
    'weights' : ['uniform' ,'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1,2]
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
# 최적의 하이퍼파라미터 및 정확도 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최적의 교차 검증 점수 (평균 정확도): {:.2f}".format(grid_search.best_score_))

# 테스트 데이터에서의 성능 평가
test_accuracy = grid_search.score(X_test, y_test)
print("테스트 세트에서의 정확도: {:.2f}".format(test_accuracy))
```
>**출력결과**
>
>최적의 하이퍼파라미터: {'metric': 'manhattan', 'n_neighbors': 1, 'p': 1, 'weights': 'distance'}
>
>최적의 교차 검증 점수 (평균 정확도): 0.88
>
>테스트 세트에서의 정확도: 0.90

**최종 parameter 결정**

metric : manhattan, p 사용 X(minkowski를 사용할 때 필요한 변수이기 때문), weight: distance

n_neighbors: 이 변수에 대한 고민이 많았다. 수가 적을수록 train set에 적합하겠지만 과적합이 되어 test set에 적합하지 않을 수 있고 수가 클수록 편향이 되어 정확도 값이 높지 않을 것이기 때문이다. round2에 제출했던 코드에서 나는 n_neighbors를 3이라고 지정했을 때 0.74가 나왔으므로 2나 4로 정하려고 했고 
  - **최종 모델**
