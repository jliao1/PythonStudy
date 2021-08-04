
#--第1种：-----------------------------------------------------
def compare(self, a, b):
    if a + b > b + a:
        return -1
    return 1

array = [1, 20, 23, 4, 8]
string = []
# 把整型转换成字符串
for i in array:
    string.append(str(i))

import functools
string.sort(key=functools.cmp_to_key(compare))


#--第2种：(比1更简洁点)----------------------------------------------------
nums = [1, 20, 23, 4, 8]
from functools import cmp_to_key
nums.sort(key=cmp_to_key(lambda x, y: 1 if str(x) + str(y) < str(y) + str(x) else -1))


#--第3种：----------------------------------------------------
class NumStr:
    def __init__(self, v):
        self.v = v
    def __lt__(self, other):
        return self.v + other.v < other.v + self.v

A = ["12", "121"]
A.sort(key = NumStr)
