import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
#  데이터셋을 엑셀 파일에서 로드
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
excel_file_path = "S_features_30_sec.csv"

# GTZAN 데이터셋 로드
X, y = load_gtzan_dataset_csv(excel_file_path)

# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Z 점수 계산
z_scores = stats.zscore(X_train_scaled)

# Z 점수가 특정 임계값을 초과하는 데이터 포인트 식별
threshold = 1
outliers = (np.abs(z_scores) > threshold).all(axis=1)
num_outliers = np.sum(outliers)
print("=======================================")
print(f"Number of outliers: {num_outliers}")
# 이상치 제거
X_train_no_outliers = X_train_scaled[~outliers]
y_train_no_outliers = y_train[~outliers]

# KNN 모델 초기화
knn_model = KNeighborsClassifier()

param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}


grid_search = GridSearchCV(knn_model, param_grid, scoring='f1_micro', cv=10)



# 그리드 서치를 사용하여 최적의 하이퍼파라미터 찾기
grid_search.fit(X_train_no_outliers, y_train_no_outliers)

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 테스트 세트에서 예측
y_pred = grid_search.predict(X_test_scaled)

# F1 score 출력
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score on Test Set:", f1)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")