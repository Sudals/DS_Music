from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
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

pca_kernel = 'linear'
svm_kernel = 'rbf'
max_iter=800
# 음악 파일에서 특징 추출
excel_file_path = "features_3_sec.csv"

X, y = load_gtzan_dataset_csv(excel_file_path)


ss = StandardScaler()
X_Scale = ss.fit_transform(X)

svm_model_T = SVC(kernel=svm_kernel, probability=True)

# 차원 범위 설정 (예: 1부터 55까지)
n_components_range = list(range(2, 58))

best_score = 0
best_n_components = 54

# # 차원 범위에서 가장 높은 정확도를 갖는 차원 탐색
for n_components in n_components_range:
    kpca = KernelPCA(n_components=n_components,kernel=pca_kernel)
    X_kpca = kpca.fit_transform(X_Scale)
    scores = cross_val_score(svm_model_T, X_kpca, y, cv=5)
    mean_score = scores.mean()
    print(f"{n_components} : {mean_score}")
    if mean_score > best_score:
        best_score = mean_score
        best_n_components = n_components

# 최적 차원 출력
print("Best Dimension:", best_n_components)

# 최적 차원으로 변환
best_kpca = KernelPCA(n_components=best_n_components,kernel=pca_kernel)
X_best_kpca = best_kpca.fit_transform(X_Scale)
joblib.dump(best_kpca, 'kpca_model.pkl')
print(X_Scale.shape[1])
# 데이터 분할 (학습용 데이터와 테스트용 데이터로 분리)
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.2,random_state=25)



# t-SNE를 사용하여 2차원으로 시각화
tsne = TSNE(n_components=2, random_state=25)
X_tsne = tsne.fit_transform(X_train)

label_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'black', 'grey']

# 시각화
# plt.figure(figsize=(15, 10))
# for i, label in enumerate(range(1, 11)):
#     indices = (y_train == label)
#     plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],c=label_colors[i], label=f'Label {label}')
#
# plt.title(f'Label')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend()
# plt.tight_layout()
# plt.show()


# # 시각화
# plt.figure(figsize=(15, 10))
# for i, label in enumerate(range(1, 11)):
#     indices = (y == label)
#     plt.subplot(2, 5, i + 1)
#     plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],c=label_colors[i], label=f'Label {label}')
#     plt.title(f'Label')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.legend()
#
#
# plt.tight_layout()
# plt.show()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# SVM 모델 생성
svm_model = SVC(kernel=svm_kernel,probability=True)
# grid_search = GridSearchCV(svm_model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# best_C = grid_search.best_params_['C']
# best_svm_model = grid_search.best_estimator_
# print(f"Best C Value: {best_C}")
# y_pred = best_svm_model.predict(X_test)
# report = classification_report(y_test, y_pred)
# print(report)
# 다른 커널 옵션: 'rbf' (RBF 커널), 'poly' (다항식 커널) 등
# 모델 학습
svm_model.fit(X_train, y_train)

mean = ss.mean_
std = ss.scale_
print(len(mean))
np.savetxt('mean.txt', mean, fmt='%.22f')
np.savetxt('std.txt', std, fmt='%.18f')
# 테스트 데이터에 대한 예측
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(svm_model.score(X_train,y_train))
print(svm_model.score(X_test,y_test))
print(report)
# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)

# 훈련 데이터에 대한 판별 함수 값
decision_values = svm_model.decision_function(X_train)

print(f"Accuracy: {accuracy:.2f}")
print(y_pred)
variances = X_train.var(axis=0)

# 감마 값을 계산 (n_features * X.var())
gamma_value = 1 / (55 * variances.mean())
print(f"Estimated Gamma Value: {gamma_value}")
cv_results = cross_validate(svm_model, X_best_kpca, y, cv=5, return_train_score=True)
train_scores = cv_results['train_score']
print("Train scores:", train_scores)
print("Average train score:", np.mean(train_scores))
test_scores = cv_results['test_score']
print("Test scores:", test_scores)
print("Average test score:", np.mean(test_scores))
model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
# 클래스별 Precision-Recall을 계산합니다.
precision = dict()
recall = dict()

plt.figure(figsize=(8, 6))

# for i in range(1, 10):  # 클래스 수에 맞게 수정
#     # One-vs-Rest 방식으로 이진 분류기 생성
#     svm_binary = SVC(kernel='rbf', probability=True)
#
#     # 클래스 i를 양성 클래스로 설정하고 나머지 클래스를 음성 클래스로 설정
#     y_train_binary = (y_train == i).astype(int)
#     y_test_binary = (y_test == i).astype(int)
#
#     # 모델 학습
#     svm_binary.fit(X_train, y_train_binary)
#
#     # 클래스 i에 대한 예측 확률 계산
#     y_scores = svm_binary.predict_proba(X_test)[:, 1]  # 양성 클래스의 확률만 사용
#
#     # Precision-Recall 곡선 계산
#     precision, recall, _ = precision_recall_curve(y_test_binary, y_scores)
#
#     # 곡선 그리기
#     plt.plot(recall, precision, lw=2, label='class {}'.format(i))
#
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="best")
# plt.title("Precision-Recall Curve for each class (One-vs-Rest)")
# plt.grid(True)
# plt.show()
