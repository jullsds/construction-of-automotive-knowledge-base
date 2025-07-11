import pandas as pd

df = pd.read_csv('车质网汽车投诉.csv')
# 提取“典型问题”列
typical_problems = df['典型问题']

# 使用空格为分界符拆分数据，并展开为多行
split_data = typical_problems.str.split(' ', expand=True).stack()

# 将拆分后的数据转换为新的 dataframe
new_df = pd.DataFrame(split_data, columns=['拆分后的问题'])

# 统计每个问题的出现次数
problem_counts = new_df['拆分后的问题'].value_counts().reset_index()
problem_counts.columns = ['拆分后的问题', '出现次数']

# 去重
new_df = new_df.drop_duplicates()

# 合并出现次数到去重后的 dataframe
new_df = new_df.merge(problem_counts, on='拆分后的问题', how='left')

# 按“出现次数”列由高到低排序
new_df = new_df.sort_values(by='出现次数', ascending=False)

# 以“|”为分隔符拆分“拆分后的问题”列，并分为两列
split_columns = new_df['拆分后的问题'].str.split('|', expand=True)

# 将拆分后的两列分别命名为“二级分类”和“三级分类”
new_df['二级分类'] = split_columns[0]
new_df['三级分类'] = split_columns[1]

# 删除原始的“拆分后的问题”列
new_df = new_df.drop(columns=['拆分后的问题'])

# 重置索引
new_df = new_df.reset_index(drop=True)

# 保存结果到CSV文件# 在导入部分添加os模块
import os

# ...（原有代码保持不变）...

# 创建存放分类文件的目录
os.makedirs('分类结果', exist_ok=True)

# 按二级分类分组并保存为多个CSV文件
for category, group in new_df.groupby('二级分类'):
    # 生成安全的文件名（替换非法字符）
    safe_filename = category.replace('|', '_').replace('\\', '_').replace('/', '_')
    file_path = f'分类结果/{safe_filename}.csv'

    # 保存该分类的数据（包含二级分类和三级分类列）
    group[['二级分类', '三级分类']].to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f'已保存: {file_path}')
