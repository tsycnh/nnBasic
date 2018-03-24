#=====如何import=======
import time

print(time.localtime())

#===== # 自定义名称
import time as t # 自定义名称
print(t.time())

#=====导入指定功能
from time import time,localtime
print(localtime())
print(time())

#====导入所有功能
from time import *
print(clock())
