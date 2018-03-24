#====if===========
a =1
b =1
c=3

if a >b:
    print('a 大于b')

'''
if True:
    do sth.
'''

if a<b<c:
    print('a<b<c')
'''
< > <= >=
== !=
'''
if a!=b:
    print('a!=b')

#=if=else=============
if a>b:
    print('a>b')
else:# 注意缩进
    print('a<=b')

#=if=elif=else==========
if a >1:
    print('a>1')
elif a==1:
    print('a==1')
elif a < 1:# change to 2 讲解elif
    print('a<1')
else:
    print('else')


