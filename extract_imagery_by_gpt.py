import openai
import pandas as pd
import re
import os
import random
from http import HTTPStatus
from dashscope import Generation
import dashscope
base_dir = './result'
save_dir = './qwen-result'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
file_names = os.listdir(base_dir)

openai.api_type = "azure"
openai.api_base = "https://test-azureopenai-003.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "1a825d951cb7464b85b593126e3a6134"
dashscope.api_key = 'sk-667dbe0bc8f74a0ba832a2b0611f2d08'

def get_break_point(save_path):
    paths = os.listdir(save_path)
    if not len(paths)==0:
        num = []
        for p in paths:
            num.append(int(p[:-4]))
        num = sorted(num)
        return num[-1]
    else:
        return 0

def call_with_messages(messages):
    response = Generation.call(model="qwen-max-0428",
                               messages=messages,
                               # 将输出设置为"message"格式
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
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
    你是一个名词提取助手，我将给你一些词语和对应的解释，你将从解释中抽取出不带有任何修饰成分的名词，比如月亮、太阳、风、花
    注意不要出现任何修饰成分，不要出现‘xx的xx’
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

st_point = get_break_point(save_dir)
for i in range(st_point,len(file_names)):
    file_name = f'{i+1}.txt'
    print(file_name)
    file_path = base_dir + f'/{file_name}'
    save_path = save_dir + f'/{file_name}'
    with open(file_path,'r') as file:
        with open(save_path,'w') as save_file:
            lines = file.readlines()
            mes = '\n'.join(lines)
            messages[-1]['content'] = mes
            # res = get_completion_from_messages(messages)
            res = call_with_messages(messages)
            print(res)
            save_file.write(res)
