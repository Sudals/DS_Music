import librosa
import pandas as pd
import os
import numpy as np

def analyze_audio(file_path, label, index, duration=30):
    # 음원 파일 로드
    y, sr = librosa.load(file_path, sr=22050)

    print(f"샘플링 속도: {sr} Hz")  # 샘플링 속도
    print(f"Audio Length: {y.shape}")  # 음원 전체 길이

    segment_duration = 30  # 구간 길이
    step = 1  # 이동할 스텝
    j=0
    max_energy = 0
    max_start = 0
    tmp_segment=0
    for start in range(0, len(y) - segment_duration * sr, step * sr):
        segment = y[start:start + segment_duration * sr]  # 30초 세그먼트 자르기
        energy = sum(abs(segment)) / len(segment)  # 에너지의 평균 계산

        if energy > max_energy:
            max_energy = energy
            tmp_segment=segment
            max_start = start / sr  # 최대 에너지를 가진 구간의 시작 시간

    #y=tmp_segment
    segments = []
    max_start_int = int(max_start)
    for start in range(max_start_int, max_start_int + duration * sr, 3 * sr):
        segment = y[start:start + 3 * sr]

        chroma_stft_mean = librosa.feature.chroma_stft(y=segment).mean()
        chroma_stft_var = librosa.feature.chroma_stft(y=segment).var()
        rms_mean = librosa.feature.rms(y=segment).mean()
        rms_var = librosa.feature.rms(y=segment).var()
        spectral_centroid_mean = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
        spectral_centroid_var = librosa.feature.spectral_centroid(y=segment, sr=sr).var()
        spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean()
        spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y=segment, sr=sr).var()
        rolloff_mean = librosa.feature.spectral_rolloff(y=segment, sr=sr).mean()
        rolloff_var = librosa.feature.spectral_rolloff(y=segment, sr=sr).var()
        zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(segment).mean()
        zero_crossing_rate_var = librosa.feature.zero_crossing_rate(segment).var()

        # 추가적인 특징
        harmony, perceptr = librosa.effects.hpss(segment)
        harmony_mean, harmony_var = harmony.mean(), harmony.var()
        perceptr_mean, perceptr_var = perceptr.mean(), perceptr.var()

        # onset 강도 계산 및 템포 추출
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # MFCC 특징 (20개 계수)
        mfcc_features = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        mfcc_means = mfcc_features.mean(axis=1)
        mfcc_vars = mfcc_features.var(axis=1)
        # print("=============")
        # print(start+3*sr)
        # print(j)
        fsa=(start+3*sr)/(j+1)
        # print(fsa)
        # print(max_start_int)
        # print(j)
        segment_features = [f"{label}.{str(index).zfill(5)}.{j}.wav",fsa, chroma_stft_mean.mean(), chroma_stft_var.mean(),
                    rms_mean.mean(), rms_var.mean(), spectral_centroid_mean.mean(), spectral_centroid_var.mean(),
                    spectral_bandwidth_mean.mean(), spectral_bandwidth_var.mean(), rolloff_mean.mean(),
                    rolloff_var.mean(),
                    zero_crossing_rate_mean.mean(), zero_crossing_rate_var.mean(),
                    harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo]

        for i in range(len(mfcc_means)):
            segment_features.append(mfcc_means[i])
            segment_features.append(mfcc_vars[i])
        segment_features.append(label)


        segments.append(segment_features)
        j+=1
    df = pd.DataFrame(segments, columns=[
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
    return df


def map_data(file_path, label, i):
    return [f"{label}.{str(i).zfill(5)}.wav",os.path.basename(file_path),label]


def save_to_csv(dataframes, output_path):

    # Save DataFrame to CSV
    dataframes.to_csv(output_path, index=False, encoding='cp949')




def save_to_mapping(data, output_mapping_path):
    # 데이터프레임 생성 및 CSV로 저장
    df = pd.DataFrame(data, columns=[
        'filename', 'orgname', 'label'
    ])
    # csv 파일로 저장 (인코딩은 cp949로 설정)
    df.to_csv(output_mapping_path, index=False, encoding='cp949')


if __name__ == "__main__":
    labelClass = "reggae"  # 장르

    input_directory = os.getcwd() + "/SoundTrack/" + labelClass  # 현재 경로 가져오기
    print(input_directory)
    output_csv_path = "out_Data.csv"  # 출력 csv 이름 설정
    output_mapping_path = "out_Map.csv"

    data = pd.DataFrame()  # csv에 쓰일 data 리스트
    map = []
    i = 100  # 음원 인덱스

    for filename in os.listdir(input_directory):  # 폴더 내부 탐색
        if filename.endswith(".mp3") or filename.endswith(".wav"):  # 파일이 mp3 or wav인 경우
            file_path = os.path.join(input_directory, filename)  # path 합치기
            print(filename)

            # analyze_audio 함수를 통해 특징 추출
            features = analyze_audio(file_path, labelClass, i)
            mapping = map_data(file_path, labelClass, i)

            data=pd.concat([data,features], axis=0)  # 결과를 data 리스트에 저장
            #print(data)

            map.append(mapping)
            i += 1  # 음원 인덱스 증가

    # 추출된 특징들을 csv 파일로 저장
    #print(map[0])

    save_to_csv(data, output_csv_path)
    save_to_mapping(map,output_mapping_path)
