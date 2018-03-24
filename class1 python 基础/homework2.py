#  班级管理系统

'''
功能：
=. 录入。输入学生信息，保存到管理系统
=.* 删除。删除某个指定学生
=.* 学生信息修改
=. 查看所有学生
=. 本地保存和读取
=. 面向对象编程
=. 多情况处理
'''

class Student:
    def __init__(self,num,name,sex,age):
        self.num = num
        self.name = name
        self.sex = sex
        self.age = age
    def format(self):
        text = self.num + ','+ self.name+','+self.sex+','+self.age+'\n'
        return text

Looping = True
filename = 'm.csv'
while Looping:
    status =  input("你想做什么？\na:增加新学生,d:删除学生,m:修改学生信息,c:查看所有学生信息,x:退出\n请输入: ")

    if status == 'a':
        info = input('请输入学生信息,格式如下，注意各信息用空格分割\n学号 姓名 性别 年龄\n请输入：')
        d = info.split(' ')
        new_student = Student(d[0],d[1],d[2],d[3])
        my_file = open(filename,'a')
        my_file.write(new_student.format())
        my_file.close()

    elif status == 'd':
        pass
    elif status == 'm':
        pass
    elif status == 'c':
        pass
    elif status == 'x':
        Looping = False
    else:
        print("输入错误，请重新输入")

print('结束')
