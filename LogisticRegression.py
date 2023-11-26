import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def removeOutlier(train_scaled, train_target):
    # 각 특성의 Z-score 계산
    z_scores = stats.zscore(train_scaled)

    # Z-score가 특정 값을 넘는 데이터 제거
    threshold = 2
    filtered_indices = np.all(np.abs(z_scores) < threshold, axis=1)

    # 이상치가 제거된 데이터셋
    return train_scaled[filtered_indices], train_target[filtered_indices]

def showResult(count = 100):
    predictions = lr.predict(test_scaled[:count])
    answer = test_target[:count]

    # 예측 값과 실제 정답 출력
    for i in range(count):
        print(f"예측: {predictions[i]}, 정답: {answer[i]}")



music = pd.read_csv('Data/features_30_sec_1.csv')

music_input = music[['chroma_stft_mean', 'chroma_stft_var', 'chroma_stft_var', 'rms_var', 'spectral_centroid_mean',
'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean',
'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean',
'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean',
'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean',
'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']].to_numpy()

music_target = music['label'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(music_input, music_target, test_size=0.2)

ss = StandardScaler()
ss.fit(train_input)

# 데이터 정규화
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 이상치 제거
# train_scaled, train_target = removeOutlier(train_scaled, train_target)

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print("우리팀 데이터1 정확도 : " + str(lr.score(test_scaled, test_target)))

# proba = lr.predict_proba(test_scaled[:5])
# print(lr.classes_)
# print(np.round(proba, decimals=3))

music2 = pd.read_csv('Data/features_30_sec_2.csv')

music_input2 = music2[['chroma_stft_mean', 'chroma_stft_var', 'chroma_stft_var', 'rms_var', 'spectral_centroid_mean',
'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean',
'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean',
'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean',
'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean',
'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']].to_numpy()

music_target2 = music2['label'].to_numpy()

train_input2, test_input2, train_target2, test_target2 = train_test_split(music_input2, music_target2, test_size=0.2)

ss.fit(train_input2)

# 데이터 정규화
train_scaled2 = ss.transform(train_input2)
test_scaled2 = ss.transform(test_input2)

print("우리팀 데이터2 정확도 : " + str(lr.score(test_scaled2, test_target2)))

# showResult(100)
