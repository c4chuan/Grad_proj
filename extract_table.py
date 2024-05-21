# 文件路径
file_path = r"E:\localRepository\Grad_proj\data_standard.docx"
import re
import win32com.client
from docx import Document

def is_table_begin(table):
    first_row = table.rows[0].cells
    for cell in first_row:
        if len(re.findall('\d',cell.text))>0:
            return False
    return True

def read_doc(file_path):
    word = win32com.client.Dispatch("Word.Application")
    doc = word.Documents.Open(file_path)
    tables = []
    for table in doc.Tables:
        table_data = []
        print(table)
        for row in table.Rows:
            row_data = []
            for cell in row.Cells:
                row_data.append(cell.Range.Text.strip())
            table_data.append(row_data)
        tables.append(table_data)
    doc.Close()
    word.Quit()
    tables = read_doc(file_path)
    for table in tables:
        for row in table:
            print(row)
    return tables


def read_table_from_word(file_path):
    # 加载现有的Word文档
    doc = Document(file_path)
    with open('./data_standard_tables.txt','w',encoding='utf-8') as file:
        # 读取文档中的所有表格
        for i, table in enumerate(doc.tables):
            if is_table_begin(table):
               file.write('$\n')
            print(f"Table {i}:")
            for row in table.rows:
                for cell in row.cells:
                    file.write(cell.text)
                    file.write(' | ')
                    print(cell.text, end=" | ")
                file.write('\n')
                print()  # 每一行结束后换行



# 调用函数，输出Word文件中表格数据
read_table_from_word('data_standard.docx')

# print(re.findall('\d','07 骨料'))
