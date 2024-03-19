from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
print(torch.version)
# 设置模型
model_path = '../llm/chatglm3-6b/ChatGLM3/THUDM/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

# 设置要获取的文件
data_path = './sql_auto.xlsx'
df = pd.read_excel(data_path)
answer = []

# 得到回复
for i in range(len(df)):
    context = df['contexts'][i]
    query = df['question'][i]
    message = f'''
给定一段上下文{context}，请你根据上下文，回答下面的问题，不要输出除了答案之外的字符，并且答案一定不能有"引号
问题：{query}
    '''
    response,_ = model.chat(tokenizer,message,history=[])
    answer.append(response)

df['answer'] = answer
df.to_excel('result_sql_auto_top3_glm.xlsx')


