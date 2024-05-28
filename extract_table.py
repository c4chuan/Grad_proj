import re
import win32com.client
from docx import Document

def is_table_begin(table):
    first_row = table.rows[0].cells
    for cell in first_row:
        if len(re.findall('\d',cell.text))>0:
            return False
    return True
def read_table_from_word(file_path,out_put_path):
    # 加载现有的Word文档
    doc = Document(file_path)
    with open(out_put_path,'w',encoding='utf-8') as file:
        # 读取文档中的所有表格
        for i, table in enumerate(doc.tables):
            if is_table_begin(table):
               file.write('$\n')
            print(f"Table {i}:")
            for row in table.rows:
                for cell in row.cells:
                    text = cell.text.replace('\n','')
                    file.write(text)
                    file.write(' | ')
                    print(cell.text, end=" | ")
                file.write('\n')
                print()  # 每一行结束后换行

def check_with_content(content,k):
    if len(content)>k:
        lines = content.split('\n')
        head = lines[0]
        result = []
        content_length = len(head)+2
        temp_result = [head]
        for line in lines[1:]:
            if content_length+len(line)+2 <k:
                temp_result.append(line)
                content_length+=len(line)+2
            else:
                result.append('\n'.join(temp_result))
                temp_result = [head]
                content_length = len(head)+2
        result.append('\n'.join(temp_result))
        return result
    else:
        return [content]
def split_table_with_k(path,result_path,k=1000):
    result = []
    with open(path,'r',encoding='utf-8') as file:
        contents = file.read().split('$\n')
        for content in contents:
            result+=check_with_content(content,k)
        with open(result_path,'w',encoding='utf-8') as save_file:
            save_file.write('\n$\n'.join(result))

# 调用函数提取出文档中的所有表格
file_path = 'data_standard_fixed.docx'
out_put_path = './data_standard_tables.txt'
result_path = './new_data_standard_tables_fixed.txt'

read_table_from_word(file_path,out_put_path)

# 表格中有超出切分字符的部分，所以设定k=1000，将表格按最多1000字符切分开
split_table_with_k(out_put_path,result_path,k=1000)
# print(re.findall('\d','07 骨料'))
