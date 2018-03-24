#======类基础===========
class Person:# 首字母大写（约定成俗）
    # 属性
    name = "Xiao Ming"
    age = 30
    # 方法
    def eat(self,food):# 类内方法的第一个参数都会指向当前的类
        print(self.name,food)
        self.food = food
        print('name:',self.name)

# 实例化

a_person = Person()
print(a_person.name)
a_person.eat('apple')
print(a_person.food)
a_person.eat('banana')
print(a_person.food)
