# 验证码生成器
# 生成200个不重复的6位验证码，由数字组成
# 存储至文件
# 思路1：随机生成一个六位数即可
# 思路2：生成六个随机的一位数，拼接即可
# 要求+：不止数字，还有字符（ABC...)
import random

text = ""
all_nums = []
for i in range(0,200):
    verify_num = random.randint(100,300)
    while verify_num in all_nums:
        verify_num = random.randint(100,300)
        print('new num')
    
    
##    print(verify_num)
    text+=str(verify_num)+'\n'
    all_nums.append(verify_num)

print(text)
my_file = open('verify num.txt','w')
my_file.write(text)
my_file.close()


