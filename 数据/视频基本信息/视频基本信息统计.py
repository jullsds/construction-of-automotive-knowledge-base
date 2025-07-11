import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取多个Excel文件
file_paths = ['FM老常315.xlsx', '拜托了老司机.xlsx', '晓北-城市私家车.xlsx']  # 替换为实际路径
dfs = [pd.read_excel(f) for f in file_paths]
df = pd.concat(dfs, ignore_index=True)

# 2. 数据预处理
df = df[['点赞', '评论', '转发', '收藏']].dropna()  # 选择目标列并清除空值

# 3. 区间分布统计函数
def count_intervals(series):
    bins = [0, 100, 500, 1000, 5000, 10000, series.max()]
    labels = ['0-100', '101-500', '501-1k', '1k-5k', '5k-10k', '10k+']
    return pd.cut(series, bins=bins, labels=labels, right=False).value_counts().sort_index()

# 4. 统计各指标分布
metrics = ['点赞', '评论', '转发', '收藏']
distributions = {metric: count_intervals(df[metric]) for metric in metrics}

# 5. 可视化展示
plt.figure(figsize=(15,10))
colors = sns.color_palette("husl", 4)

for i, metric in enumerate(metrics, 1):
    plt.subplot(2,2,i)
    distributions[metric].plot(kind='bar', color=colors[i-1])
    plt.title(f'{metric}分布区间', fontsize=12)
    plt.xlabel('区间范围', fontsize=10)
    plt.ylabel('视频数量', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')

plt.tight_layout()
plt.show()
