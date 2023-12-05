import matplotlib.pyplot as plt
import numpy as np

classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
attribute1_values = [0.82,
0.72,
0.78,
0.78,
0.64,
0.88,
0.7,
0.68,
0.71
]
attribute2_values = [0.79,
0.56,
0.71,
0.76,
0.62,
0.87,
0.73,
0.63,
0.65
]
attribute3_values = [0.81,
0.51,
0.63,
0.7,
0.45,
0.87,
0.68,
0.48,
0.67
]
attribute4_values = [0.42,
0.5,
0.67,
0.6,
0.54,
0.87,
0.47,
0.43,
0.67
]

# 막대 그래프 그리기
bar_width = 0.2
index = np.arange(len(classes))

fig, ax = plt.subplots()
bar1 = ax.bar(index - 1.5*bar_width, attribute1_values, bar_width, label='RBF')
bar2 = ax.bar(index - 0.5*bar_width, attribute2_values, bar_width, label='Linear')
bar3 = ax.bar(index + 0.5*bar_width, attribute3_values, bar_width, label='Poly')
bar4 = ax.bar(index + 1.5*bar_width, attribute4_values, bar_width, label='Sigmoid')

# 그래프에 레이블과 범례 추가
ax.set_xlabel('Classes')
ax.set_ylabel('Attribute Values')
ax.set_title('Comparison of 4 Attributes for 9 Classes')
ax.set_xticks(index)
ax.set_xticklabels(classes)
ax.legend()

# 그래프 표시
plt.show()