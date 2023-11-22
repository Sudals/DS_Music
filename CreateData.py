import librosa
import pandas as pd
import os


def analyze_audio(file_path, label,  i, duration=30):
    # 음원 파일 로드
    y, sr = librosa.load(file_path, sr=22050)

    print(f"샘플링 속도: {sr} Hz")  # 샘플링 속도
    print(f"Audio Length: {y.shape}")  # 음원 전체 길이

    # 에너지가 높은 구간 찾기
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    # 에너지가 높은 구간의 시작 지점을 찾아 사용할 duration 설정
    if len(onset_frames) > 0:
        start_frame = onset_frames[0]
        y = y[start_frame:start_frame + int(sr * duration)]

    # 각종 특징 추출
    chroma_stft_mean = librosa.feature.chroma_stft(y=y).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y).var()

    # 추가적인 특징
    harmony, perceptr = librosa.effects.hpss(y)
    harmony_mean, harmony_var = harmony.mean(), harmony.var()
    perceptr_mean, perceptr_var = perceptr.mean(), perceptr.var()

    # onset 강도 계산 및 템포 추출
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # MFCC 특징 (20개 계수)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = mfcc_features.mean(axis=1)
    mfcc_vars = mfcc_features.var(axis=1)
    #print(mfcc_means)
    #print(mfcc_vars)

    # 특징들을 리스트로 반환
    features = [f"{label}.{str(i).zfill(5)}.wav", len(y), chroma_stft_mean.mean(), chroma_stft_var.mean(),
                rms_mean.mean(), rms_var.mean(), spectral_centroid_mean.mean(), spectral_centroid_var.mean(),
                spectral_bandwidth_mean.mean(), spectral_bandwidth_var.mean(), rolloff_mean.mean(), rolloff_var.mean(),
                zero_crossing_rate_mean.mean(), zero_crossing_rate_var.mean(),
                harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo]

    for i in range(len(mfcc_means)):
        features.append(mfcc_means[i])
        features.append(mfcc_vars[i])
    features.append(label)
    return features


def map_data(file_path,label,i):
    return [f"{label}.{str(i).zfill(5)}.wav",os.path.basename(file_path),label]


def save_to_csv(data, output_path):
    # 데이터프레임 생성 및 CSV로 저장
    df = pd.DataFrame(data, columns=[
        'filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
        'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var',
        'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
        'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var',
        'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
        'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
        'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
        'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 'label'
    ])
    # csv 파일로 저장 (인코딩은 cp949로 설정)
    df.to_csv(output_path, index=False, encoding='cp949')


def save_to_mapping(data, output_mapping_path):
    # 데이터프레임 생성 및 CSV로 저장
    df = pd.DataFrame(data, columns=[
        'filename', 'orgname', 'label'
    ])
    # csv 파일로 저장 (인코딩은 cp949로 설정)
    df.to_csv(output_mapping_path, index=False, encoding='cp949')


if __name__ == "__main__":
    # 현재 작업 디렉토리 내의 모든 음악 파일 분석
    labelClass = "disco" # 장르
    input_directory = os.getcwd() + "/SoundTrack/" + labelClass  # 음원 경로 가져오기
    output_csv_path = "output_data.csv"  # 출력 csv 이름 설정
    output_mapping_path = "output_mapping.csv"
    data = []  # csv에 쓰일 data 리스트
    map = []
    i = 100  # 음원 인덱스 - 추가로 음원 데이터셋 추가 시 기존에 있던 인덱스 이후의 인덱스로 수정해서 써야합니다.

    for filename in os.listdir(input_directory):  # 폴더 내부 탐색
        if filename.endswith(".mp3") or filename.endswith(".wav"):  # 파일이 mp3 or wav인 경우
            file_path = os.path.join(input_directory, filename)  # path 합치기

            # analyze_audio 함수를 통해 특징 추출
            features = analyze_audio(file_path, labelClass, i)
            mapping = map_data(file_path, labelClass, i)

            data.append(features)  # 결과를 data 리스트에 저장

            map.append(mapping)
            i += 1  # 음원 인덱스 증가

    # 추출된 특징들을 csv 파일로 저장
    #print(map[0])
    save_to_csv(data, output_csv_path)
    save_to_mapping(map, output_mapping_path)
