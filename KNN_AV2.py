import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.decomposition import PCA
import joblib
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
excel_file_path = "result/2.csv"

# GTZAN 데이터셋 로드
X, y = load_gtzan_dataset_csv(excel_file_path)

scaler = StandardScaler()
#mean=[3982.105259083702, 1976836.2706275356, 0.0731954897310237, 0.0014917932585568074, 6.13349280998751e-05, 0.050155487956679234, 2.170167269288383e-05, 0.011858261843052428, 122.79749742058611, -78.78239765025593, 2778.629652053683, 111.7203561017479, 766.0240252584258, 0.21274313363171032, 379.3651279662916, 29.17464940424466, 177.35889726779058, 5.071347069304616, 119.71668656491866, 11.36219605767478, 96.90342561860162, 0.6154663614483769, 81.83019666978768, 6.341908038196008, 71.74838240761548, -1.016142874172597, 66.44399940043691, 3.8418605261785297, 62.259243367665356, -1.8934303849159162, 60.43406424531836, 3.0618617791734684, 57.85195346691642, -2.059976881615853, 57.905859119850184, 0.6521998091593632, 54.989231121722845, -2.744408999289634, 54.35979354182274, 1.3682237501586776, 51.965959350187326, -2.7321717997663573, 51.17691022097377, 1.2846625538309004, 52.784461373907604, -1.981652125831127, 55.25414702559311, 0.25063613611610475]
#std=[1436.1755662346552, 1763196.1606705838, 0.02818377333106165, 0.0017468211020369147, 0.002283484362859017, 0.025172499166085386, 0.0008627104953317433, 0.00948329833924365, 22.613421988689197, 78.3181712040126, 2353.2446699800894, 34.03622418626518, 637.6429175624238, 23.09983033666116, 330.03184577593777, 13.227857208121062, 129.26025089067008, 8.793497303064832, 94.14645419799179, 7.531921179915747, 71.60824600934649, 5.96767828574818, 52.60266864143732, 7.195251274767478, 44.48063810594593, 5.7299904615770965, 37.9358797555558, 6.07512800211857, 36.68054742259353, 5.052881737077977, 36.03322557525221, 5.1821042887385165, 29.801275780779797, 4.48423227734252, 31.860278711017944, 4.762263998215127, 30.278493136361718, 4.218688873707293, 31.171499546485524, 4.296614463484276, 31.224710456091632, 3.910805328120573, 32.81452515114798, 4.031820487685561, 31.91628032036015, 3.602883375468, 31.250084330994937, 4.079735372017705]

#X_train_scaled = (X_train-mean)/std
#X_test_scaled = (X_test-mean)/std
X_train_scaled = scaler.fit_transform(X)
mean = scaler.mean_
std = scaler.scale_
np.savetxt('mean.txt', mean, fmt='%.22f')
np.savetxt('std.txt', std, fmt='%.18f')


# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=0.2,random_state=42)


# 추정된 평균과 표준 편차를 파일에 저장
#np.savetxt('mean.txt', mean)
#np.savetxt('std.txt', std)
# mean = scaler.mean_
# std = scaler.scale_
# mean = mean.tolist()
# mean_d = [float(num) for num in mean[6:-1]]
# std = std.tolist()
# std_d = [float(num) for num in std[6:-1]]
#
# # 변환된 값을 txt 파일에 저장
# with open('mean_std.txt', 'w') as f:
#     f.write('mean: ' + str(mean_d) + '\n')
#     f.write('std: ' + str(std_d) + '\n')

# KNN 모델 초기화
knn_model = KNeighborsClassifier()

param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}


grid_search = GridSearchCV(knn_model, param_grid, scoring='f1_micro', cv=5)



# 그리드 서치를 사용하여 최적의 하이퍼파라미터 찾기
grid_search.fit(X_train,y_train)

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 테스트 세트에서 예측
y_pred = grid_search.predict(X_test)

# F1 score 출력
f1 = f1_score(y_test, y_pred, average='micro')
print("F1 Score on Test Set:", f1)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
model_filename = 'knn_model.pkl'
#joblib.dump(grid_search, model_filename)

