import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import joblib
from scipy import stats
def convert_labels_to_strings(labels):
    label_mapping = {
        1: '클래식',
        2: '컨트리',
        3: '디스코',
        4: '힙합',
        5: '재즈',
        6: '메탈',
        7: '팝',
        8: '레게',
        9: '락'
    }

    converted_labels = [label_mapping[label] for label in labels]
    return converted_labels
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

    return X.to_numpy()
loaded_kpca = joblib.load('kpca_model.pkl')
mean = np.loadtxt('mean.txt')
std = np.loadtxt('std.txt')
print(len(mean))
# 음악 파일에서 특징 추출
excel_file_path = "result/TestFile.csv"
X = load_gtzan_dataset_csv(excel_file_path)

scaler = StandardScaler()
#print(len(X))
X_test_scaled = (X-mean)/std

# 모델 불러오기
loaded_model = joblib.load('svm_model.pkl')
print("=================")
X_test_pca = loaded_kpca.transform(X_test_scaled)
# 예측
prediction = loaded_model.predict(X_test_pca)
# accuracy = accuracy_score(y, prediction)
# print(f"Accuracy: {accuracy}")
# report = classification_report(y, prediction)
# print(report)
predicted_probabilities = loaded_model.predict_proba(X_test_pca)

# 임계값 설정
threshold = 0.3

for i, sample_probs in enumerate(predicted_probabilities):
    # 확률이 큰 순서대로 라벨의 인덱스를 정렬
    sorted_labels = np.argsort(sample_probs)[::-1]

    # 임계값을 초과하는 라벨 찾기
    selected_labels = sorted_labels[sample_probs[sorted_labels] > threshold]
    if len(selected_labels)!=0 :
        converted_labels = convert_labels_to_strings(selected_labels + 1)
        print(f"Sample {i + 1} - Selected Labels (Sorted):", converted_labels)
    else :
        print(f"Sample {i + 1} - Selected Labels (Sorted): 장르없음",)
# 출력된 확률을 확인합니다.
print(predicted_probabilities)
most_common_value = np.argmax(np.bincount(prediction))
print(most_common_value)
print(prediction)
np.savetxt('predictions.txt', prediction, fmt='%d')
