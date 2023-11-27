import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

music = pd.read_csv('features_30_sec_2.csv')

music_input = music[features].to_numpy()
music_target = music['label'].to_numpy()

# 데이터 정규화
ss = StandardScaler()
ss.fit(music_input)

music_scaled = ss.transform(music_input)

with open('model/logisticRegressionModel.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

print("정확도 : " + str(loaded_model.score(music_scaled, music_target)))
