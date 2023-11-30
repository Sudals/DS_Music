import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import MDS

warnings.filterwarnings("ignore", message="The default value of `normalized_stress` will change")

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
        'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var',
        'tempo',
        'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var',
        'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var',
        'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
        'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var',
        'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
        'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
        'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var',
        'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var',
        'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
        'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
    ]

    # 선택한 특징 열과 레이블 열 선택
    df_selected = df[selected_feature_cols + ['label']]

    # X는 선택한 특징 열, y는 'genre' 열
    X = df_selected.drop(columns=['label'])

    y = df_selected['label']


    return X.to_numpy(), y.to_numpy()

# 음악 파일에서 특징 추출
excel_file_path = "result/total_3sec/DS_feature_3_sec_1.csv"

# 데이터셋 로드
X, y = load_gtzan_dataset_csv(excel_file_path)

# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN 모델 초기화
knn_model = KNeighborsClassifier()

param_grid = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}

# 훈련 데이터의 연결성 행렬 계산
connectivity = kneighbors_graph(X_train_scaled, n_neighbors=10, mode='connectivity')
grid_search = GridSearchCV(knn_model, param_grid, scoring='f1_micro', cv=5)

# 그리드 서치를 사용하여 최적의 하이퍼파라미터 찾기
grid_search.fit(X_train_scaled, y_train)

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 테스트 세트에서 예측
y_pred = grid_search.predict(X_test_scaled)
# 거리 행렬이 대칭인지 확인하고, 대칭이 아니라면 대칭으로 만들기
distances = (connectivity + connectivity.T) / 2

# 희소 행렬을 밀집 행렬로 변환
distances_dense = distances.toarray()

# MDS를 사용하여 거리 행렬을 2차원으로 축소
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
X_2d = mds.fit_transform(distances_dense)

# 각 레이블 값에 대한 산점도 그리기
plt.figure(figsize=(12, 8))
for label in np.unique(y_train):
    indices = (y_train == label)
    plt.scatter(X_2d[indices, 0], X_2d[indices, 1], label=f'Label {label}')

plt.title('2D Embedding of Data Points Based on Nearest Neighbors Distances')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.show()
# F1 score 출력
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score on Test Set:", f1)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")