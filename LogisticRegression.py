import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate
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


# def showResult(count=100):
#     #predictions = lr.predict(test_scaled[:count])
#     answer = test_target[:count]
#
#     # 예측 값과 실제 정답 출력
#     for i in range(count):
#         print(f"예측: {predictions[i]}, 정답: {answer[i]}")


features = ['chroma_stft_mean', 'chroma_stft_var', 'chroma_stft_var', 'rms_var', 'spectral_centroid_mean',
            'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean',
            'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean',
            'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean',
            'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
            'mfcc13_mean',
            'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var',
            'mfcc17_mean',
            'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']

# music = pd.read_csv('result/GTZAN_features_30_sec.csv')
music = pd.read_csv('result/features_30_sec_single_label9.csv')
# music = pd.read_csv('features_3_sec.csv')

music_input = music[features].to_numpy()
music_target = music['label'].to_numpy()
ss = StandardScaler()
X=ss.fit_transform(music_input)

train_input, test_input, train_target, test_target = train_test_split(X, music_target, test_size=0.2,random_state=25)


mean = ss.mean_
std = ss.scale_

np.savetxt('mean_logistic.txt', mean, fmt='%.22f')
np.savetxt('std_logistic.txt', std, fmt='%.18f')

# 이상치 제거
# train_scaled, train_target = removeOutlier(train_scaled, train_target)

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_input, train_target)

with open('model/logisticRegressionModel.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

print("학습 데이터 정확도 : " + str(lr.score(train_input, train_target)))
print("테스트 데이터 정확도 : " + str(lr.score(test_input, test_target)))
cv_results = cross_validate(lr, X, music_target, cv=5, return_train_score=True)
train_scores = cv_results['train_score']
print("Train scores:", train_scores)
print("Average train score:", np.mean(train_scores))
test_scores = cv_results['test_score']
print("Test scores:", test_scores)
print("Average test score:", np.mean(test_scores))
# proba = lr.predict_proba(test_scaled[:5])
# print(lr.classes_)
# print(np.round(proba, decimals=3))

# showResult(100)
