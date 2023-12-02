from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import KernelPCA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import joblib
def load_gtzan_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    missing_values = df.isnull().sum()
    print(missing_values)

    # 결측치가 있는 열 제거 또는 다른 방법으로 처리
    df = df.dropna()
    # NaN 값을 평균값으로 대체
    nan_values = df.isnull().sum()
    print(nan_values)
    df = df.fillna(0)
    rows_with_nan = df[df.isnull().any(axis=1)]

    print("Rows with NaN values:")
    print(rows_with_nan)
    # infinity 값을 대체하거나 해당 행 제거
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    # 수동으로 선택한 특징 열
    # selected_feature_cols = [
    #     'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    #     'spectral_centroid_mean',  'spectral_bandwidth_mean',
    #     'rolloff_mean',  'zero_crossing_rate_mean',
    #      'harmony_var', 'perceptr_var', 'tempo',
    #     'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var',
    #     'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
    #     'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var',
    #     'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
    #     'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
    #     'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
    #     'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
    # ]
    selected_feature_cols = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
        'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var',
        'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
        'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var',
        'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
        'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
        'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
        'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
    ]
    print(len(selected_feature_cols))
    # 선택한 특징 열과 레이블 열 선택
    df_selected = df[selected_feature_cols + ['label']]

    # X는 선택한 특징 열, y는 'genre' 열
    X = df_selected.drop(columns=['label'])
    y = df_selected['label']

    return X.to_numpy(), y.to_numpy()

pca_kernel = 'poly'
svm_kernel = 'rbf'
# 음악 파일에서 특징 추출
excel_file_path = "result/features_30_sec_single_label9.csv"

X, y = load_gtzan_dataset_csv(excel_file_path)


ss = StandardScaler()
X_Scale = ss.fit_transform(X)

svm_model_T = SVC(kernel=svm_kernel, probability=True)


# Isomap 및 SVM을 통합하여 성능을 평가하는 함수
def evaluate_isomap_svm(X_train, X_test, y_train, y_test, n_neighbors, C, n_components):
    # Isomap을 사용하여 차원을 감소
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_train_isomap = isomap.fit_transform(X_train)
    X_test_isomap = isomap.transform(X_test)

    # SVM 모델 초기화 및 훈련
    svm_model = SVC(kernel='linear', C=C)
    svm_model.fit(X_train_isomap, y_train)

    # 모델 평가
    y_pred = svm_model.predict(X_test_isomap)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# 교차 검증을 통한 그리디 방법으로 최적 파라미터 찾기
def grid_search(X, y, n_neighbors_range, C_range, n_components_range):
    param_grid = {'isomap__n_neighbors': n_neighbors_range,
                  'svm__C': C_range,
                  'isomap__n_components': n_components_range}

    pipeline = Pipeline([
        ('isomap', Isomap()),
        ('svm', SVC(kernel='linear'))
    ])

    isomap_svm = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5  # 교차 검증 폴드 수
    )

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 훈련 및 최적 파라미터 찾기
    isomap_svm.fit(X_train, y_train)

    # 최적 파라미터 및 정확도 출력
    best_params = isomap_svm.best_params_
    best_accuracy = isomap_svm.best_score_
    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_accuracy)

    return best_params,best_accuracy

# 예시로 n_neighbors, C, n_components의 범위 설정
n_neighbors_range = [5]
C_range = [1]
n_components_range = list(range(2,57))

# 최적 파라미터 찾기
best_params,best_accuracy = grid_search(X, y, n_neighbors_range, C_range, n_components_range)
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)
# X_train, X_test, y_train, y_test = train_test_split(X_best_ism, y, test_size=0.2,random_state=25)
#
#
#
# # t-SNE를 사용하여 2차원으로 시각화
# tsne = TSNE(n_components=2, random_state=25)
# X_tsne = tsne.fit_transform(X_train)
#
# label_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'black', 'grey']
#
# # 시각화
# # plt.figure(figsize=(15, 10))
# # for i, label in enumerate(range(1, 11)):
# #     indices = (y_train == label)
# #     plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],c=label_colors[i], label=f'Label {label}')
# #
# # plt.title(f'Label')
# # plt.xlabel('t-SNE Dimension 1')
# # plt.ylabel('t-SNE Dimension 2')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
#
#
# # # 시각화
# # plt.figure(figsize=(15, 10))
# # for i, label in enumerate(range(1, 11)):
# #     indices = (y == label)
# #     plt.subplot(2, 5, i + 1)
# #     plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],c=label_colors[i], label=f'Label {label}')
# #     plt.title(f'Label')
# #     plt.xlabel('t-SNE Dimension 1')
# #     plt.ylabel('t-SNE Dimension 2')
# #     plt.legend()
# #
# #
# # plt.tight_layout()
# # plt.show()
# # SVM 모델 생성
# svm_model = SVC(kernel=svm_kernel,probability=True)
# # 다른 커널 옵션: 'rbf' (RBF 커널), 'poly' (다항식 커널) 등
# # 모델 학습
# svm_model.fit(X_train, y_train)
#
# mean = ss.mean_
# std = ss.scale_
# print(len(mean))
# np.savetxt('mean.txt', mean, fmt='%.22f')
# np.savetxt('std.txt', std, fmt='%.18f')
# # 테스트 데이터에 대한 예측
# y_pred = svm_model.predict(X_test)
# report = classification_report(y_test, y_pred)
# print(svm_model.score(X_train,y_train))
# print(svm_model.score(X_test,y_test))
# print(report)
# # 정확도 평가
# accuracy = accuracy_score(y_test, y_pred)
#
# print(f"Accuracy: {accuracy:.2f}")
# print(y_pred)
# variances = X_train.var(axis=0)
#
# # 감마 값을 계산 (n_features * X.var())
# gamma_value = 1 / (55 * variances.mean())
# print(f"Estimated Gamma Value: {gamma_value}")
# cv_results = cross_validate(svm_model, X_best_kpca, y, cv=5, return_train_score=True)
# train_scores = cv_results['train_score']
# print("Train scores:", train_scores)
# print("Average train score:", np.mean(train_scores))
# test_scores = cv_results['test_score']
# print("Test scores:", test_scores)
# print("Average test score:", np.mean(test_scores))
# model_filename = 'svm_model.pkl'
# joblib.dump(svm_model, model_filename)
