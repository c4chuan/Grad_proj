import openai
import pandas as pd
import re
import os
base_dir = './result'
file_names = os.listdir(base_dir)

openai.api_type = "azure"
openai.api_base = "https://test-azureopenai-003.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "1a825d951cb7464b85b593126e3a6134"


def get_completion_from_messages(messages, temperature=0):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messages,
        temperature=temperature  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_data_from_response(res):
    query =re.findall(r'problem: (.+?)？',res)
    gt = re.findall(r'answer: (.+?)。',res)
    print(res)
    print(gt)
    print(len(gt))
    return query, gt

messages = [
    {'role': 'system', 'content': f'''
    你是一个名词提取助手，我将给你一些词语和对应的解释，你将从解释中抽取出不带有任何修饰成分的名词
    输入的格式为：
    1.【<词语>】: <解释>
    2.【<词语>】: <解释>
    3.【<词语>】: <解释>
    ...
    输出格式必须严格遵循以下格式：
    1.【<词语>】: <名词>
    2.【<词语>】: <名词>
    3.【<词语>】: <名词>
    ...
    10.【<词语>】: <名词>
    '''},
    {'role': 'user', 'content': ''}]
for file_name in file_names:
    print(file_name)
    file_path = base_dir + f'/{file_name}'
    with open(file_path,'r') as file:
        lines = file.readlines()
        mes = '\n'.join(lines)
        messages[-1]['content'] = mes
        res = get_completion_from_messages(messages)
        print(res)