# # 各二级分类数量统计
# import pandas as pd
#
# df = pd.read_csv('车质网汽车投诉.csv')
#
# typical_problems = df['典型问题']
#
# # 使用空格为分界符拆分数据，并展开为多行
# split_data = typical_problems.str.split(' ', expand=True).stack()
#
# # 将拆分后的数据转换为新的 dataframe
# new_df = pd.DataFrame(split_data, columns=['拆分后的问题'])
#
# # 以“|”为分隔符拆分“拆分后的问题”列，并分为两列
# split_columns = new_df['拆分后的问题'].str.split('|', expand=True)
#
# # 将拆分后的两列分别命名为“二级分类”和“三级分类”
# new_df['二级分类'] = split_columns[0]
# new_df['三级分类'] = split_columns[1]
#
# # 统计二级分类数量
# category_counts = new_df['二级分类'].value_counts().reset_index()
# category_counts.columns = ['二级分类', '数量']  # 列重命名
#
# # 按数量降序排序
# category_counts = category_counts.sort_values(by='数量', ascending=False)
#
# # 显示结果
# print("\n二级分类统计结果：")
# print(category_counts)


import pandas as pd

# 读取CSV文件（注意处理BOM字符）
file_path = "车质网汽车投诉_处理结果.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 统计每个二级分类出现的次数
category_counts = df['二级分类'].value_counts().reset_index()
category_counts.columns = ['二级分类', '出现次数']  # 重命名列

# 按出现次数降序排序
category_counts = category_counts.sort_values('出现次数', ascending=False)

# 打印结果
print("二级分类统计结果（按出现次数排序）：")
print(category_counts.to_string(index=False))

# 可选：保存结果到新CSV
# category_counts.to_csv("二级分类次数统计.csv", index=False, encoding='utf-8-sig')
