# 0-- 输入用户文本及权重
user_text = input("请输入您的文本：")
# "2025年6月27日（在唐山市南新道和友谊路口发生重大交通事故），导致车辆左前脸车辆正面撞击安全气囊未弹出，左车门后移打不开，很明显气囊应该起到作用，与4S店厂家产生争议，与4s店协商多次，厂家置之不理，没有答复，无人过问（当时我们主动刹车，正面撞击力非常大，安全气囊未打开，未能及时保护车主的安全，导致车主头部急速磕到方向盘上，拿受害者生命当试验品）质量问题，厂家不认同，怕砸了自己的品牌，为了保护自己的品牌厂家不敢承认，多次撇清车辆质量问题，安全气囊的作用不就是保护人员安全的吗？不但不起作用，反倒以欺骗手段，欺骗更多的消费者！这样谁还敢拿自己的生命开玩笑！人命关天！请有关部门严查！"
w1 = float(input("请输入内容完整性权重："))
w2 = float(input("请输入内容准确性权重："))
w3 = float(input("请输入内容规范性权重："))
w4 = float(input("请输入人物情感真实性权重："))
w5 = float(input("请输入基础信息权重："))
# 定义权重（根据实际需求调整）
weights = {
    '内容完整性评分': w1,
    '内容准确性评分': w2,
    '内容规范性评分': w3,
    '人物情感真实性评分': w4,
    '基础信息评分': w5
}
# 确保权重总和为1（可选检查）
if abs(sum(weights.values()) - 1.0) > 0.01:
    print(f"警告: 权重总和为{sum(weights.values()):.2f}，不是100%")


# 1-- 运行工作流
from cozepy import COZE_CN_BASE_URL

coze_api_token = 'pat_cNaEpnPvcbdcLyaePTVTnEZgqbIL5absjxIinp4PEmOtlY1pelHnwOTvsDcT7Hs4'
coze_api_base = COZE_CN_BASE_URL
from cozepy import Coze, TokenAuth, Message, ChatStatus, MessageContentType  # noqa
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)
workflow_id = '7468947507445678089'


workflow = coze.workflows.runs.create(
    workflow_id='7494253667928735770',
    parameters={
        "input": user_text
        }
)
# print("workflow.data", workflow.data)


#2-- 工作流返回数据处理
import pandas as pd
import json
from ast import literal_eval

# 将workflow.data转换为字符串
response_str = str(workflow.data)

# # 打印转换后的字符串（可选）
# print("转换后的字符串:")
# print(response_str)

# 将字符串解析为Python字典
try:
    response_data = json.loads(response_str)
except json.JSONDecodeError:
    # 如果JSON解析失败，使用ast安全解析
    try:
        response_data = literal_eval(response_str)
    except:
        print("无法解析响应数据")
        response_data = {}

# 提取URL列表数据
url_list = response_data.get("url", [])

# 准备存储解析后的数据
parsed_data = []

# 遍历每个URL条目
for item in url_list:
    # 解析fields字段中的JSON字符串
    try:
        # 尝试直接解析JSON
        fields = json.loads(item["fields"])
    except:
        # 如果JSON解析失败，使用ast安全解析
        try:
            fields = literal_eval(item["fields"])
        except:
            # 如果解析失败，跳过此项
            continue

    # 准备当前条目的数据
    entry = {}

    # 处理投诉问题（取第一个文本元素）
    if "投诉问题" in fields:
        complaint_list = fields["投诉问题"]
        if isinstance(complaint_list, list) and len(complaint_list) > 0:
            entry["投诉问题"] = complaint_list[0].get("text", "")

    # 处理作品网址（取第一个文本元素）
    if "作品网址" in fields:
        url_list = fields["作品网址"]
        if isinstance(url_list, list) and len(url_list) > 0:
            entry["作品网址"] = url_list[0].get("text", "")

    # 处理评分字段（直接取值）
    for score_field in ["内容完整性评分", "内容准确性评分", "内容规范性评分", "人物情感真实性评分"]:
        if score_field in fields:
            # 确保评分是数值类型
            try:
                entry[score_field] = float(fields[score_field])
            except (ValueError, TypeError):
                entry[score_field] = None

    # 处理互动数据字段（点赞、评论、收藏、转发）
    for interaction_field in ["点赞", "评论", "收藏", "转发"]:
        if interaction_field in fields:
            interaction_list = fields[interaction_field]
            if isinstance(interaction_list, list) and len(interaction_list) > 0:
                # 尝试转换为整数
                try:
                    entry[interaction_field] = int(interaction_list[0].get("text", ""))
                except (ValueError, TypeError):
                    entry[interaction_field] = interaction_list[0].get("text", "")

    # 添加record_id作为唯一标识（可选）
    entry["record_id"] = item["record_id"]

    # 添加到结果列表
    parsed_data.append(entry)

# 创建DataFrame
df = pd.DataFrame(parsed_data)

# 重新排列列顺序（按需求）
desired_columns = [
    "投诉问题", "作品网址", "内容完整性评分", "内容准确性评分",
    "内容规范性评分", "人物情感真实性评分",
    "点赞", "评论", "收藏", "转发", "record_id"
]

# 只保留存在的列
available_columns = [col for col in desired_columns if col in df.columns]
df = df[available_columns]

# # 打印结果
# print(df)


# 3-- 相似度计算
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 本地加载模型
model_path = "models/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


user_emb = get_sentence_embedding(user_text)
def custom_function(text):
    emb = get_sentence_embedding(text)
    cos_sim = F.cosine_similarity(user_emb, emb)
    return format(cos_sim.item(), ".4f")

# 计算相似度（用户文本与所有投诉问题）
df['相似度'] = df['投诉问题'].apply(custom_function)
df_sorted = df.sort_values(by='相似度', ascending=False)

# df_sorted.to_csv('相似度计算结果.csv', index=False, encoding='utf-8-sig')


# 4-- 基础评分计算
# 使用 copy() 创建副本，避免在切片上操作
df_top5 = df_sorted.head(5).copy()

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
df_top5.loc[:, '_点赞评分'] = df_top5['点赞'].apply(lambda x: map_score(x, [500, 1000, 5000, 10000]))
df_top5.loc[:, '_评论评分'] = df_top5['评论'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))
df_top5.loc[:, '_收藏评分'] = df_top5['收藏'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))
df_top5.loc[:, '_转发评分'] = df_top5['转发'].apply(lambda x: map_score(x, [100, 500, 1000, 5000]))

# 计算基础信息评分（四列平均值）
df_top5.loc[:, '基础信息评分'] = (df_top5['_点赞评分'] + df_top5['_评论评分'] +
                             df_top5['_收藏评分'] + df_top5['_转发评分']) / 4

# 删除不再需要的列（原始数据和中间评分列）
columns_to_drop = ['点赞', '评论', '收藏', '转发', 'record_id', '_点赞评分', '_评论评分', '_收藏评分', '_转发评分']
df_top5.drop(columns=columns_to_drop, inplace=True, errors="ignore")


# 5-- 实现推荐
# 计算加权总分
df_top5.loc[:, '总分'] = 0  # 初始化总分列
for column, weight in weights.items():
    if column in df_top5.columns:
        df_top5.loc[:, '总分'] += df_top5[column] * weight

# 按总分降序排序
df_final = df_top5.sort_values(by='总分', ascending=False)

print("\n根据您的输入，为您找到以下三个视频：")
for index, row in df_final.head(3).iterrows():
    print(row['作品网址'])

