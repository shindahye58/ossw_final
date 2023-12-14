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
# SVM 모델
svm = SVC()

# 탐색할 하이퍼파라미터 그리드
param_grid = {
    'C': [15, 20, 25],  # C 값 후보
    'kernel': ['poly', 'sigmoid'],  # 커널 후보
    'gamma': ['scale', 0.1, 1, 10],  # gamma 값 후보
    'degree' : [2,3,4],
    'coef0' : [0, 0.5, 1, 2]
}

# Grid Search를 통해 최적의 하이퍼파라미터 탐색
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최적의 교차 검증 점수 (평균 정확도): {:.2f}".format(grid_search.best_score_))

# 테스트 데이터에서의 성능 평가
test_accuracy = grid_search.score(X_test, y_test)
print("테스트 세트에서의 정확도: {:.2f}".format(test_accuracy))
```

  - **model tuning**
