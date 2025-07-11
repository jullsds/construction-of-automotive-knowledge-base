import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 设置matplotlib的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建存储词云图的文件夹，使用绝对路径
output_dir = os.path.join(os.getcwd(), '词云图结果')
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"文件夹已成功创建: {output_dir}")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

df = pd.read_csv('车质网汽车投诉.csv')
# 提取“典型问题”列
typical_problems = df['典型问题']

# 使用空格为分界符拆分数据，并展开为多行
split_data = typical_problems.str.split(' ', expand=True).stack()

# 将拆分后的数据转换为新的 dataframe
new_df = pd.DataFrame(split_data, columns=['拆分后的问题'])

# 以“|”为分隔符拆分“拆分后的问题”列，并分为两列
split_columns = new_df['拆分后的问题'].str.split('|', expand=True)

# 将拆分后的两列分别命名为“二级分类”和“三级分类”
new_df['二级分类'] = split_columns[0]
new_df['三级分类'] = split_columns[1]

# 删除原始的“拆分后的问题”列
new_df = new_df.drop(columns=['拆分后的问题'])

# 重置索引
new_df = new_df.reset_index(drop=True)

# 按“二级分类”进行分组
grouped = new_df.groupby('二级分类')

# 对每个分组中的“三级分类”绘制词云图
for name, group in grouped:
    text = ' '.join(group['三级分类'].dropna())
    wordcloud = WordCloud(
        font_path='simhei.ttf', 
        width=800, 
        height=400, 
        background_color='white',
        max_words=50,  # 减少最大词语数量，避免留白
        min_font_size=20,  # 增加最小字体大小，使词语更显眼
        collocations=False,  # 避免重复词语
        prefer_horizontal=0.9,  # 增加水平排列的词语比例
        scale=2  # 增加图像缩放比例，使词语更紧凑
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # 替换文件名中的非法字符
    safe_name = name.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f'{safe_name}.png')
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"词云图已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存词云图时出错: {e}")
    plt.close()  # 关闭图像以释放内存
