#===========文件写入==============

text = "good morning.\ngood night.\n"
print(text)
# docs https://docs.python.org/3/
my_file = open('this file.txt','w')
my_file.write(text)
my_file.close()
#===========文件追加写入===========
new_text = "good tomorrow\n"
my_file = open('this file.txt','a')
my_file.write(new_text)
my_file.close()
#===========文件读取===============
my_file = open('this file.txt','r')# 读入的是文件
content = my_file.read() # 读入文件所有内容
print(type(my_file))
content_line = my_file.readline()# 每调用一次读取一行
content_lines = my_file.readlines()# 读取所有内容，按照行放入list中

print(content_line)
