# FINAL PROJECT

**< Brain Tumor Classification using by method in scikit learn package >**

*my Evaluation environment*

>Numpy 버전: 1.26.2
>
>scikit learn 버전: 1.3.2
>
>scikit image 버전: 0.22.0

## Using model
> K-Nearest Neighbors(KNN): 

## training model process

**Optimize hyper-parameters**

  - **model's hyper-parameters**

  - **grid search**

    >grid search란?  머신 러닝에서 모델의 최적 하이퍼파라미터를 찾기 위한 기술 중 하나,  이는 여러 가지 하이퍼파라미터 조합을 시도하여 모델의 성능을 평가하고, 최적의 조합을 찾는 과정
    >
    > 하이퍼파라미터 그리드 생성 🠒 모델 훈련 및 검증 🠒 최적 하이퍼 파라미터 선택 🠒 모델 평가
    >
```
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1, 3, 5],# 이웃 수에 대한 후보 값
    'weights' : ['uniform', 'distance'],
    'metric': ['euclidean', 'minkowski'],
    'algorithm': ['auto']
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

  - **model tuning**
