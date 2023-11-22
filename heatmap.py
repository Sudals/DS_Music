import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터프레임 생성
# CSV 파일 읽기
data = pd.read_csv('Data/features_30_sec.csv')

# 첫 번째 열과 마지막 열 제외하고 나머지 열 선택
selected_data = data.iloc[:, 1:-1]  # 첫 번째 열부터 마지막 열 이전까지 선택
df = pd.DataFrame(selected_data)

# 상관관계 계산
corr = df.corr()

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()