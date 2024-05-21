import codecs
import os

# 文件所在目录
file_path = "./result"
files = os.listdir(file_path)

for file in files:
    print(file)
    file_name = file_path + '/' + file
    f = codecs.open(file_name, 'r', 'gbk')
    ff = f.read()
    file_object = codecs.open(file_path + '/' + file, 'w', 'utf-8')
    file_object.write(ff)