
class Abc:#类
    name='good'#属性

    def __init__(self):# 构造函数
        print('构造函数被调用了')
        print(self.name)#访问类内的函数和属性
        self.good()
    def good(self):
        print('god')
        self.name
        self.name = '777'


a = Abc()  # 生成实例
print(a.name)
b = Abc()