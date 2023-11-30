from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
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
    selected_feature_cols = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var','spectral_centroid_mean',
        'spectral_centroid_var',
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
excel_file_path = "result/features_30_sec_1.csv"

X, y = load_gtzan_dataset_csv(excel_file_path)


ss = StandardScaler()
X_Scale = ss.fit_transform(X)

# 데이터 분할 (학습용 데이터와 테스트용 데이터로 분리)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(kernel='rbf', probability=True))
])
param_grid = {
    'pca__n_components': list(range(40,55))
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)  # 5-fold 교차 검증 사용
grid_search.fit(X_train, y_train)

print("Best PCA Components:", grid_search.best_params_['pca__n_components'])
# 최적의 PCA 구성 요소로 변환
best_pca = grid_search.best_estimator_.named_steps['pca']

# X_train과 X_test를 최적의 PCA 구성 요소로 변환
X_train_pca = best_pca.transform(X_train)
X_test_pca = best_pca.transform(X_test)
best_scaler = grid_search.best_estimator_.named_steps['scaler']
X_train_scaled = best_scaler.transform(X_train_pca)  # 데이터 스케일링
X_test_scaled = best_scaler.transform(X_test_pca)  # 데이터 스케일링
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)  # PCA로 변환된 데이터로 모델을 학습

# 테스트 데이터로 예측
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best PCA components: {accuracy:.2f}")


best_scaler_mean = best_scaler.mean_  # 스케일러의 평균
best_scaler_scale = best_scaler.scale_
np.savetxt('mean.txt', best_scaler_mean, fmt='%.22f')
np.savetxt('std.txt', best_scaler_scale, fmt='%.18f')

model_filename = 'svmp_model.pkl'
joblib.dump(grid_search, model_filename)

