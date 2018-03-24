# 变量的作用域
# 层级,内部可以访问外部，外部不能访问内部

A = 3

def func():
    a = 666
    global B# 强制在函数内部定义全局变量
    B = 'bbb'
    print(A)

func()
print(B)


