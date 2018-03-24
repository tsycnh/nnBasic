#=======tuple===list=====
# 元组和列表都是一连串有顺序的数字
#元组tuple

a_tuple = (2,4,1,565,33)
b_tuple = 5,7,8,1,1,11,3

a_list = [3,56,67,1,1,6,6]

#取值 []索引
print(a_tuple[0])
print(a_list[1])
a_list[0] = 4
##a_tuple[0] = 8 # tuple 不可更改
# 循环输出
for item in a_tuple:
    print(item)

for item in a_list:
    print(item)

for i in range(len(a_list)):
    print(i,a_list[i])

# 另一种循环输出enumerate
print('====================')
for i,item in enumerate(a_list):
    print(i,item)

#====list列表=================
print('====================')

#append 在末尾添加
print(a_list)
a_list.append(999)
print(a_list)

#insert 在指定位置添加
a_list.insert(1,666)# 在第1的位置添加666，后面依次顺延
print(a_list)

#remove 移除

a_list.remove(1)# 移除第一次出现值为1的数
print(a_list)

#index索引
print(a_list[0])
print(a_list[-1])
print(a_list[1:4])#[3, 666, 56, 67, 1, 6, 999]
#output? []
print(a_list[4:])
print(a_list[-2:])

# 打印索引
print(a_list.index(666))# 第一次出现666的index是多少
print(a_list.count(6))# 计算6出现的次数
# 排序
a_list.sort()
print(a_list)
a_list.sort(reverse=True)
print(a_list)
