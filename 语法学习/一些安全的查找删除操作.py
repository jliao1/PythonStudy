"""
查找某个字串/字符
str = 'abcdef'
index = str.find('abc')  如果不存在 返回 -1， 如果存在返回 sub字串的第一次出现的index

这种会报错：
s = 'a'
int(s)  # 因为s根本就不是数字，无法转换成数字


集合的 添加，删除，查找，取长度是O(1)
遍历是 O(n)
删除元素，如果用remove找不到会报错。用discard比较安全，找不到就找不到，set remains unchanged

字典dict
get(key[, default])
Return the value for key if key is in the dictionary, else default.
If default is not given, it defaults to None, so that this method never raises a KeyError.
用下标取value要先判断，key在不在
如果不在，就会报错
所以一般还是用get取，get(key, default_value) 如果找不到key就会返回default value


pop(key[, default])
If key is in the dictionary, remove it and return its value, else return default.
If default is not given and key is not in the dictionary, a KeyError is raised.

setdefault(key[, default])
If key is in the dictionary, return its value.
If not, insert key with a value of default and return default. default defaults to None.



"""

