"""
【String Split】
好像string的slice时间复杂度是O(n)
两种方法分割python多空格字符串
    str = " aa   bbbbb         ccc  d"

    这种不行：
    str_list2 = str.split(' ')
    print(str_list2)  # 打印出来是 ['', 'aa', '', '', 'bbbbb', '', '', '', '', '', '', '', '', 'ccc', '', 'd']

    这种可以：
    str_list1 = str.split()
    print(str_list1)  # 打印出来是 ['aa', 'bbbbb', 'ccc', 'd']


【排序】
系统内置的是stable的
string类型的是不能sort的，list可以
（1）
    # char是个list，出来的结果是小写在前，大写在后
    chars.sort(key=lambda c: c.isupper())
（2）
    intervals.sort( key=lambda pair: pair.start )
    # 上面的写法好像是直接在 intervals 上 sort，sort 完后 intervals 是不变的
    # 若写成这个也行 intervals = sorted(intervals, key=lambda pair: pair.start)
    #     但sort后的intervals 的地址就跟之前不一样了，是个新的对象了
（3）按多列优先级排序
    bids = [[2, 7, 8, 1], [3, 7, 5, 1], [4, 10, 3, 3], [1, 5, 5, 0]]
    # 第3列降序排  第4列升序排
    bids.sort(key=lambda x: (-x[2], x[3]))
    再一个例子：
    log2 = [(['g1', 'act', 'car'], 'g1 act car'), (['ab1', 'off', 'key', 'dog'], 'ab1 off key dog'), (['a8', 'act', 'zoo'], 'a8 act zoo'), (['a2', 'act', 'car'], 'a2 act car')]
    log2.sort(key = lambda x : (x[0][1:], x[0][0]) )
（4）自己定义多重排序
    例子1
    class NumStr:
        def __init__(self, v):
            self.v = v
        def __lt__(self, other):
            return self.v + other.v < other.v + self.v

    A = ["12", "121"]
    A.sort(key = NumStr)
    例子2
    logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
    def reorderLogFiles(logs):
        def get_key(log):  # log是 logs里的每一个元素
            _id, rest = log.split(maxsplit=1)
            return (0, rest, _id) if rest[0].isalpha() else (1,)
            #     key1，key2，key3  先按key1排序，再按key2排序，再按key3排序
    return sorted(logs, key=get_key)  # 返回的是个排序好的list = ['let1 art can', 'let3 art zero', 'let2 own kit dig', 'dig1 8 1 5 1', 'dig2 3 6']
    例子3
    bundle了之后再降序排列
    例子4
    username = ["joe", "maybe",  "maar"]
    timestamp = [11, 2, 3]
    dic = {}
    for i in range(len(username)):
        dic[username[i]] = timestamp[i]
    # dic = {'joe': 11, 'maybe': 2, 'maar': 3}
    ll1 = sorted(dic, key = lambda x : dic[x])   # 不能写成 ll1 = sorted(dic, key = lambda x : dic.values())
    # 出来的是 keys组成的list，但是按value排序的ll1 = ['maybe', 'maar', 'joe']
    ll2 = sorted(dic)    #  也可以写成  ll2 = sorted(dic.keys())
    # 出来的是按key排序 ll2 = ['joe', 'maar', 'maybe']
    ll3 = sorted(dic.values())
    # 出来的是values组成的list按values排序 ll3 = [2, 3, 11]
    例子5
    力扣 M 1152， 对list处理 + 字典 + 一顿骚排序


    【bundle information】
    x = [1, 2, 3, 4, 5]
    y = ['a', 'b', 'c', 'd']
    xy = zip(x, y)


    【setdefault】
    # 如果 first_word 不存在 dic 的 keys 里
    # 那就把 first_word 存进 dic，且其对应 value 设置为一个空集合 set()，有返回值也是这个空集合
    dic.setdefault(first_word, set())  # 如果 first_word 存在 dic 里，返回的是它对应的 value
    dic[first_word].add(second_word)

    setdefault(key[, default])
    If key is in the dictionary, return its value.
    If not, insert key with a value of default and return default. default defaults to None.

    【combinations】
    itertools.combinations(iterable, r)
    Return r length subsequences of elements from the input iterable.
    Combinations are emitted in lexicographic sort order. So, if the input iterable is sorted, the combination tuples will be produced in sorted order.
    Elements are treated as unique based on their position, not on their value. So if the input elements are unique, there will be no repeat values in each combination.
    比如
    from itertools import combinations
    l = ['a', 'b', 'c']
    temp = combinations(l, 2)
    l = list(temp)  出来的效果是stable的

    【dict safe pop】
     如果key不存在，删除pop 这个key也不会报错的办法：
     mydict.pop("key", None)
     找key，如果key不存在也不报错：
     get(key, default_value) 如果找不到key就会返回default value

    【不能对tuple作的操作】
    c = (0.4 , [3,5])
    c[1][0] = c[1][0] + 1
    #  下句出错 TypeError: 'tuple' object does not support item assignment
    c[0] = c[1][0] / c[1][0]

"""
import heapq

if __name__ == '__main__':
    l = [1,2,3,4]
    li=iter(l)
    print(len(li))
    print(next(li))
    print(next(li))
    print(len(li))
    print(next(li))
    print(next(li))
    print(next(li))
    print(next(li))

    import heapq

    classes = [[1,2],[3,5],[2,2]]

    ratios = [(c[0] / c[1], c) for c in classes]
    heapq.heapify(ratios)

    for _ in range(2):
        c = heapq.heappop(ratios)
        passes = c[1][0] + 1
        total = c[1][1] + 1

        heapq.heappush(ratios, (passes / total, [passes, total]))

    s = 0
    for c in ratios:
        s += c[0]


    minHeap = []
    minHeap.append([1, 5])
    minHeap.append([1, 1])
    minHeap.append([1, 2])
    minHeap.append([1, 4])
    minHeap.append([1, 3])
    heapq.heapify(minHeap)
    while len(minHeap) != 0:
        print(heapq.heappop(minHeap))
    pass
