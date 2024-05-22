import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

categories = ['Gender', 'Race', 'Region', 'LGBTQ']
metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']

data = np.array([
    [79.80, 81.13, 80.46, 79.34],  # 性别
    [78.83, 85.36, 81.96, 79.60],  # 种族
    [75.10, 81.78, 78.30, 78.44],  # 地区
    [81.48, 86.84, 84.08, 81.53]   # LGBTQ
])

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.15
bar_positions = np.arange(len(categories))

for i in range(len(metrics)):
    ax.bar(bar_positions + i * bar_width, data[:, i], bar_width, label=metrics[i])

ax.set_title('Fine-grained analysis',fontweight='bold')
ax.set_xticks(bar_positions + 1.5 * bar_width)
ax.set_xticklabels(categories)
ax.set_ylabel('preformance(%)')

ax.set_ylim(bottom=60)

ax.legend()

plt.show()
