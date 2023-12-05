from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, multilabel_confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
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
from sklearn.metrics import precision_recall_curve
import joblib
def load_gtzan_dataset_csv(file_path):
    df = pd.read_csv(file_path,skip_blank_lines=False)


    # 결측치가 있는 열 제거 또는 다른 방법으로 처리
    #df = df.dropna()
    # NaN 값을 평균값으로 대체

    # infinity 값을 대체하거나 해당 행 제거
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.dropna()
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
    df_selected = df[selected_feature_cols + ['label1', 'label2']]

    df['label2'].fillna(0, inplace=True)
    print(df['label2'].astype(int))

    y1 = df['label1']
    y1=y1.astype(int)
    y2 = df['label2']
    y2=y2.astype(int)
    for index, row in df.iterrows():

        if row['label2'] == 0:
            y2[index] = row['label1']
            #print(row)
            #print(y2[index])

    # X는 선택한 특징 열, y는 One-Hot Encoding된 label1 및 label2 열
    X = df_selected.drop(columns=['label1', 'label2'], axis=1)
    y = np.vstack((y1,y2)).T
    return X.to_numpy(), y

def load_dataset_csv(file_path):
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
# 음악 파일에서 특징 추출
excel_file_path = "result/features_30_sec_multi_label9.csv"

X, y = load_gtzan_dataset_csv(excel_file_path)
excel_file_path2 = "result/TestFile.csv"
X_test = load_dataset_csv(excel_file_path2)

ss = StandardScaler()
X_Scale = ss.fit_transform(X)
mean = ss.mean_
std = ss.scale_
print(len(mean))
np.savetxt('mean.txt', mean, fmt='%.22f')
np.savetxt('std.txt', std, fmt='%.18f')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM 분류기 생성
svm_classifier = SVC(kernel='rbf')

multioutput_classifier = MultiOutputClassifier(svm_classifier, n_jobs=-1)

# 모델 훈련
multioutput_classifier.fit(X, y)

y_pred = multioutput_classifier.predict(X_test)
print(y_pred)
# # 다중 출력을 다중 클래스-다중 라벨로 변환
# y_test_multiclass = np.argmax(y_test, axis=1)
# y_pred_multiclass = np.argmax(y_pred, axis=1)
# np.savetxt('MULTI.txt', y_test,fmt='%.0f')
# np.savetxt('MULTI2.txt', y_pred,fmt='%.0f')
# # 평가
# accuracy = accuracy_score(y_test_multiclass, y_pred_multiclass)
#
# # 결과 출력
# print("Accuracy:", accuracy)
