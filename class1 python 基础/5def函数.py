#===基本def函数（无参数，无返回值）=======
a = 1
b = 2
c = a+b
print(c)

# 函数定义
def lets_print():
    print('我在打印')
'''
def func_name(params):
    do sth.
    return sth.
'''
# 函数调用

lets_print()

# =def 带参数=============
def lets_minus(p1,p2):
    result = p1-p2
    print('lets_minus: ',result)

lets_minus(a,b)
lets_minus(3,9)
lets_minus(p1=3,p2=9)# 指定参数值,顺序无所谓
r1 = lets_minus(p2=9,p1=3)
print(r1)
# =def 返回值
def lets_minus(p1,p2):
    result = p1-p2
    print('lets_minus: ',result)
    return result

r = lets_minus(99,3)
print('r:',r)

# ==def 默认值====

def drink(name,time='早上',drinks='水'):
    print(name,'在',time,'喝了',drinks)

drink('小明','中午','牛奶')
drink('小明','中午')
drink('小明')

