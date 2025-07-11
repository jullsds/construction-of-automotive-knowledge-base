import pandas as pd

# 读取Excel文件
df = pd.read_excel('评分结果.xlsx')

# 定义评分映射函数
def map_score(value, thresholds):
    if value < thresholds[0]:
        return 1
    elif thresholds[0] <= value < thresholds[1]:
        return 2
    elif thresholds[1] <= value < thresholds[2]:
        return 3
    elif thresholds[2] <= value < thresholds[3]:
        return 4
    else:
        return 5

# 创建临时评分列（稍后删除）
df['_点赞评分'] = df['点赞'].apply(lambda x: map_score(x, [500, 1000, 5000, 10000]))
df['_评论评分'] = df['评论'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))
df['_收藏评分'] = df['收藏'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))
df['_转发评分'] = df['转发'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))

# 计算基础信息评分（四列平均值）
df['基础信息评分'] = (df['_点赞评分'] + df['_评论评分'] +
                     df['_收藏评分'] + df['_转发评分']) / 4

# 删除不再需要的列（原始数据和中间评分列）
columns_to_drop = [
    '点赞', '评论', '收藏', '转发',
    '_点赞评分', '_评论评分', '_收藏评分', '_转发评分'
]
df.drop(columns=columns_to_drop, inplace=True)

# 保存结果到新文件
df.to_excel('最终评分结果.xlsx', index=False)

print("处理完成！结果已保存到: 最终评分结果.xlsx")
print("最终保留的列:", df.columns.tolist())
