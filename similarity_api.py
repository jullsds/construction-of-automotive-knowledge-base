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

# 示例句子
emb1 = get_sentence_embedding("安全气囊亮了故障灯，经过4s店售后检修需要更换副驾驶传感器。明知道有质量问题，安全气囊各种情况都不会弹出的情况下4s店人员告知可以正常行驶。")
emb2 = get_sentence_embedding("车辆过浅坑时底盘刮蹭导致副驾驶安全气囊弹出，质疑底盘质量和安全气囊触发标准是否合理")
emb3 = get_sentence_embedding("安全气囊未弹出导致驾乘人员严重受伤（舌部断裂1/3、肋骨骨折5根），4S店推诿检测责任且厂商客服处理态度冷漠")
cos_sim = F.cosine_similarity(emb1, emb2)
print(f"文本1与文本2相似度分数: {cos_sim.item():.4f}")
print(f"文本1与文本3相似度分数: {F.cosine_similarity(emb1, emb3).item():.4f}")
print(f"文本2与文本3相似度分数: {F.cosine_similarity(emb2, emb3).item():.4f}")


# # 2. 准备需要比较的文本
# text1 = "车辆安全气囊故障，售后态度敷衍，未对解决车问题起到任何作用；自2023年7月更换一质次主板，后续接连6次进店，销售端以无故障网码无法上报厂家推脱。至今未有合理解释和对策！大众售后真不要脸，无视汽车功能安全，坚决不解决客户问题！"
# text2 = "安全气囊未弹出导致驾乘人员严重受伤，4S店推诿检测责任且厂商客服处理态度冷漠"
# text3 = "车辆在锁车熄火后，车机系统未能正常进入休眠状态，导致半夜自动播放音乐，屏幕未亮但音响工作，已发生多次，影响正常使用"
