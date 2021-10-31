"""
【heap】库必记的知识点
1. 导包写 from heapq import heapify, heappop, heappush
2. heapify时间复杂度是O(N), heappop和heappush时间复杂度是O(logN)
3. 按什么元素来计算min或max？按一个元素的列来计算，
    比如 (x，y，z) 先比x再比y再比z
4. 一个普通list要先heapify才能正常pop和push；但如果是空list，就直接pop和push了
5. 如何access最高优先级得元素？  queue[0]就可以
6. 这俩maybe好用
    List2 = heapq.nlargest(n ,List1, key=lambda x: x[1]) 从 List1 中返回前n个最大的元素组成一个新的List返回，以什么为标准排序key定
    List1 不用先 heapify，当然如果只返回1个数字的话还是用 min() max() 效率高
    List2 = heapq.nsmalles(n ,List1, key=lambda x: x[1]) 从 List1 中返回前n个最小的元素组成一个新的List返回，以什么为标准排序key定

[integer在python里不overflow]
Only floats have a hard limit in python.
Integers are are implemented as “long” integer objects of arbitrary size in python3
and do not normally overflow.

[安全删除]
set.discard(某个元素)      即使找不到也不会报错
dict.pop(key[, default])  若key存在remove it and return its value，如果找不到key就会返回 default value
        严重注意！！！如果 default is not given and key is not in the dictionary, a KeyError is raised.

[dict操作]
【取value】
确定key存在，用 dict[key] 取值
不知道key是否存在，用get取更安全，例如 dictionary.get(key, default_value) 如果找不到key就会返回 default value。
    如果找不到key又没写default value，就会返回 None. 一般还是写一下 defult value 写成空或0就可以，比图 x = d.get(key, 0) + 1
【如果key不存在，就设置它为…】
setdefault(key[, default]) If key is in the dictionary, return its value. If not, insert key with a value of default and return default. default defaults to None.
【loop】
for key, value in dict.items():
【快速用value生成list】
v = [mamp[key] for key in map]

「OrderedDict」
是按添加顺序 order 的
popitem(last=True)，像stack那样先进后出last=True，像队列那样先进先出 last=False，不写的话default是True
move_to_end(key, last=True)，移到尾部last=True，移到头部last=False，不写的话default是True；key根本就不存在的话会报错

「String操作」
str1.find(str2)  如果在str1中找得到str2，返回第一次出现的index，找不到返回 -1     而str1.index(str2)如果找不到会报错
        find() 和 index() 时间复杂度好像是 n
        ！！但要注意一个坑，如果找的是个 empty tring, 那么返回的index是0，但这往往不是我们想要的
str.isalnum()  返回真如果all characters in the string are alphanumeric

[List]
列表是可以在某个位置插入元素的
    List = [1, 2, 3]
    List.insert(1, 0)  # 插入之后是 [1, 0, 2, 3]

[小技巧]
【翻转的一些事儿】
    逆序遍历：
        for each in reversed(List):
    切片反转：
        List[::-1]
【按index sort】
    # 把英文小写字母转成 26位 array的下标
    count = [0] * 26
    count[ord(当前字母) - ord('a')] += 1 # 出现在那一位+1
【二进制和十进制互换】
    to_binary = bin(5)             # 出来是个string: '0b101'
    to_decimal = int(to_binary, 2) # 出来时个十进制的int: 5
【正则表达式】对string的提取
    import re
    paragraph = "Bob hit a221 ball123, the hit BALL flew far after it was hit."
    words1 = re.findall(r'\b[a-z]+', paragraph.lower()) # ['bob', 'hit', 'a', 'ball', 'the', 'hit', 'ball', 'flew', 'far', 'after', 'it', 'was', 'hit']
    words2 = re.findall(r'\w+', paragraph.lower())      # ['bob', 'hit', 'a2', 'ball2', 'the', 'hit', 'ball', 'flew', 'far', 'after', 'it', 'was', 'hit']
    words3 = re.findall(r'\d+', paragraph.lower())      # ['221', '123']
【python中的比较】
    l1 = [1,2,3]
    l2 = [1,2,3]
    print(l1 is l2) # False  is 比地址
    print(l1 == l2) # True   == 比值
[产生随机数字]
    import random
    num = random.randint(0, 5) 产生0-5整数







"""

