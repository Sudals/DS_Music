import chardet
import pandas as pd
def find_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())

    encoding = result['encoding']
    confidence = result['confidence']

    print(f"Detected Encoding: {encoding} with confidence: {confidence}")

    return encoding

# 파일 경로 설정
excel_file_path = "F_features_30_sec.csv"

# 파일의 인코딩 찾기
detected_encoding = find_encoding(excel_file_path)

# 찾은 인코딩으로 데이터 불러오기
try:
    df = pd.read_csv(excel_file_path, encoding=detected_encoding)
    print("Successfully loaded the file.")
except UnicodeDecodeError:
    print("Failed to load the file. Please check the encoding.")
