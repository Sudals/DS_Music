from sklearn import datasets
from sklearn.model_selection import train_test_split
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


# 음악 파일에서 특징 추출
excel_file_path = "result/features_30_sec_2.csv"

X, y = load_gtzan_dataset_csv(excel_file_path)


ss = StandardScaler()
X_Scale = ss.fit_transform(X)


print(X_Scale.shape[1])
kpca = KernelPCA(n_components=20,kernel='rbf')
X_kpca = kpca.fit_transform(X_Scale)
joblib.dump(kpca, 'kpca_model.pkl')
# 데이터 분할 (학습용 데이터와 테스트용 데이터로 분리)
X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.3,random_state=2)



# t-SNE를 사용하여 2차원으로 시각화
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_kpca)

label_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'black', 'grey']

# 시각화
plt.figure(figsize=(15, 10))
for i, label in enumerate(range(1, 11)):
    indices = (y == label)
    #plt.subplot(2, 5, i + 1)
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],c=label_colors[i], label=f'Label {label}')


plt.title(f'Label')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.tight_layout()
plt.show()
# SVM 모델 생성
svm_model = SVC(kernel='rbf',probability=True)
# 다른 커널 옵션: 'rbf' (RBF 커널), 'poly' (다항식 커널) 등
# 모델 학습
svm_model.fit(X_train, y_train)

mean = ss.mean_
std = ss.scale_
np.savetxt('mean.txt', mean, fmt='%.22f')
np.savetxt('std.txt', std, fmt='%.18f')
# 테스트 데이터에 대한 예측
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(y_pred)
variances = X_train.var(axis=0)

# 감마 값을 계산 (n_features * X.var())
gamma_value = 1 / (55 * variances.mean())
print(f"Estimated Gamma Value: {gamma_value}")
model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
