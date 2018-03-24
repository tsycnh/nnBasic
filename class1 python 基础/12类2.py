#======类基础+构造函数===========
class Person:# 首字母大写（约定成俗）
    # 属性(可删)
    name = "Xiao Ming"
    age = 30
    # 方法
    def __init__(self,name='Xiao Ming',age):# 实例初始化的时候会运行一次
        self.name = name
        self.age = age
    
    def eat(self,food):# 类内方法的第一个参数都会指向当前的类
        print(self.name,food)
        self.food = food
        print('name:',self.name)

# 实例化

laowang = Person('LaoWang',20)
print(laowang.name,laowang.age)
