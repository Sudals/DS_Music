import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터프레임 생성
# CSV 파일 읽기
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('result/features_30_sec_single_label9.csv')

# 첫 번째 열과 마지막 열 제외하고 나머지 열 선택

#selected_data = data.iloc[:, 1:-1]  # 첫 번째 열부터 마지막 열 이전까지 선택

selected_data = data.iloc[:, 1:]  # 첫 번째 열부터 마지막 열 이전까지 선택

df = pd.DataFrame(selected_data)

# 상관관계 계산
corr = df.corr()

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

#
# # VIF 계산을 위한 함수
# def calculate_vif(data):
#     vif_data = pd.DataFrame()
#     vif_data["Feature"] = data.columns
#     vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
#     return vif_data
#
# # VIF 계산
# vif_result = calculate_vif(df)

# VIF 결과에서 10을 넘는 값만 필터링하여 출력
#high_vif = vif_result[vif_result['VIF'] > 10]
#print(vif_result)


plt.show()