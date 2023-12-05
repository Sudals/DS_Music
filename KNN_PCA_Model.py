import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore", message="The default value of `normalized_stress` will change")
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
# GTZAN 데이터셋을 엑셀 파일에서 로드
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

    # 선택한 특징 열과 레이블 열 선택
    df_selected = df[selected_feature_cols + ['label']]

    # X는 선택한 특징 열, y는 'genre' 열
    X = df_selected.drop(columns=['label'])

    y = df_selected['label']


    return X.to_numpy(), y.to_numpy()

# 음악 파일에서 특징 추출
excel_file_path = "result/features_30_sec_single_label9.csv"

# 데이터셋 로드
X, y = load_gtzan_dataset_csv(excel_file_path)
ss= StandardScaler()
X_Scale = ss.fit_transform(X)

knn_model_T = KNeighborsClassifier()
n_components_range = list(range(2, 58))

best_score = 0
best_n_components = 0

# 차원 범위에서 가장 높은 정확도를 갖는 차원 탐색
for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_Scale)
    scores = cross_val_score(knn_model_T, X_pca, y, cv=5)
    mean_score = scores.mean()
    print(f"{n_components} : {mean_score}")
    if mean_score > best_score:
        best_score = mean_score
        best_n_components = n_components

# 최적 차원 출력
print("Best Dimension:", best_n_components)

# 최적 차원으로 변환
best_pca = PCA(n_components=best_n_components)
X_best_pca = best_pca.fit_transform(X_Scale)
print(X_Scale.shape[1])

# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X_best_pca, y, test_size=0.2,random_state=25)


# KNN 모델 초기화
knn_model = KNeighborsClassifier()

param_grid = {'n_neighbors': range(2,30)}

# 훈련 데이터의 연결성 행렬 계산
connectivity = kneighbors_graph(X_train, n_neighbors=10, mode='connectivity')
grid_search = GridSearchCV(knn_model, param_grid, scoring='f1_micro', cv=5)

# 그리드 서치를 사용하여 최적의 하이퍼파라미터 찾기
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 테스트 세트에서 예측
y_pred = grid_search.predict(X_test)


# F1 score 출력
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score on Test Set:", f1)
report = classification_report(y_test, y_pred)
print(grid_search.score(X_train,y_train))
print(grid_search.score(X_test,y_test))
print(report)
# 정확도 평가

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cv_results = cross_validate(grid_search, X_Scale, y, cv=5, return_train_score=True)
train_scores = cv_results['train_score']
print("Train scores:", train_scores)
print("Average train score:", np.mean(train_scores))
test_scores = cv_results['test_score']
print("Test scores:", test_scores)
print("Average test score:", np.mean(test_scores))