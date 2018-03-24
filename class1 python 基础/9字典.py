#========字典dict======
# key : value
# 字典无序
d = {'xiaoming':'23','laowang':'45'}
d1 = {1:'a',2:'b','three':'c'}
print(d,d1)
print(d1[1])
print(d1['three'])

#删除
del d1[2]
print(d1)

#增加
d1['Monday']=99
print(d1)

def func():
    print(888)
    return 8887

#可以包含各种内容
d2 = {
    'aa':[1,3,5,3],
    'bb':{'qq':999,'ww':666},
    'cc':func
    }

r = d2['cc']()
print(r)
