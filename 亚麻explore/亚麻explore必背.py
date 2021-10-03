"""
[integer在python里不overflow]
Only floats have a hard limit in python.
Integers are are implemented as “long” integer objects of arbitrary size in python3
and do not normally overflow.

「dict操作」
【取value】
确定key存在，用 dict[key] 取值
不知道key是否存在，用get取，get(key, default_value) 如果找不到key就会返回 default value
【删除key】
dict.pop(key[, default]) 若key存在remove it and return its value，如果找不到key就会返回 default value，如果 default is not given and key is not in the dictionary, a KeyError is raised.
【如果key不存在，就设置它为…】
setdefault(key[, default]) If key is in the dictionary, return its value. If not, insert key with a value of default and return default. default defaults to None.
【loop】
for key, value in dict.items():

「OrderedDict」
是按添加顺序 order 的
popitem(last=True)，像stack那样先进后出last=True，像队列那样先进先出 last=False，不写的话default是True
move_to_end(key, last=True)，移到尾部last=True，移到头部last=False，不写的话default是True；key根本就不存在的话会报错

「Set操作」
【删除一个元素】
set.discard(某个元素)  即使找不到也不会报错

「String操作」
str1.find(str2)  如果在str1中找得到str2，返回第一次出现的index，找不到返回 -1     而str1.index(str2)如果找不到会报错
find() 和 index() 时间复杂度好像是 n
！！但要注意一个坑，如果找的是个 empty tring, 那么返回的index是0，这不是我们想要的

「快速生成list」
v = [m[key] for key in m] # 用

「按位sort」
# 把英文小写字母转成 26位 array的下标
count = [0] * 26
count[ord(当前字母) - ord('a')] += 1 # 出现在那一位+1

「比等号」
    l1 = [1,2,3]
    l2 = [1,2,3]
    print(l1 is l2) # False  is 比地址
    print(l1 == l2) # True   == 比值

「正则表达式」对string的提取
import re
paragraph = "Bob hit a221 ball123, the hit BALL flew far after it was hit."
words1 = re.findall(r'\b[a-z]+', paragraph.lower()) # ['bob', 'hit', 'a', 'ball', 'the', 'hit', 'ball', 'flew', 'far', 'after', 'it', 'was', 'hit']
words2 = re.findall(r'\w+', paragraph.lower())      # ['bob', 'hit', 'a2', 'ball2', 'the', 'hit', 'ball', 'flew', 'far', 'after', 'it', 'was', 'hit']
words3 = re.findall(r'\d+', paragraph.lower())      # ['221', '123']

「string的一些公式」
str.isalnum()  返回真如果all characters in the string are alphanumeric

「二进制和十进制互换」
to_binary = bin(integer)
to_decimal = int(str,2)

{翻转的一些事儿}
逆序遍历：
    for each in reversed(List):
切片反转：
    List[::-1]

"""

