import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from collections import Counter
from sklearn.cluster import KMeans
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
excel_file_path = "result/features_30_sec_2.csv"

# GTZAN 데이터셋 로드
data, labels = load_gtzan_dataset_csv(excel_file_path)
ss = StandardScaler()
data = ss.fit_transform(data)
tsne = TSNE(n_components=2,random_state=42)
data_2d = tsne.fit_transform(data)
# K-Means 모델 생성
kmeans = KMeans(n_clusters=10,n_init=10)  # 클러스터 개수는 적절하게 설정하세요

# 데이터를 K-Means 모델에 맞게 학습시킴
kmeans.fit(data)

# 군집화된 데이터의 라벨 확인
clusters = kmeans.predict(data)
plt.figure(figsize=(8, 6))
for i in range(kmeans.n_clusters):
    cluster_data = data_2d[clusters == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')

plt.title('K-Means Clustering Visualization')
plt.legend()
plt.show()
# 각 클러스터에서 가장 많은 라벨 값을 찾음
for cluster_id in range(kmeans.n_clusters):
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
    labels_in_cluster = [labels[i] for i in cluster_indices]
    most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
    print(f"Cluster {cluster_id}: Most common label is {most_common_label}")