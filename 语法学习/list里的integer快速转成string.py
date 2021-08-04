'''
list = [3,2,1]
id1 = id(list)

list.sort()  # This method sorts the list in place

id2 = id(list)
此时 id1 和 id2 的地址是一样的

sorted(iterable, *, key=None, reverse=False)
Return a new sorted list from the items in iterable.
list2 = sorted(list)  # 此时 list 和 list2 的地址 不一样
'''

# list 里每一个是 integer
list1 = [1, 2, 3, 4]

# 这个是 convert each integer to a string
# 但它返回的是一个 iterator 迭代器
to_string = map(str, list1)
# 要这样使用
a1 = next(to_string)  # a1是字符串类型的 '1' 了
a2 = next(to_string)  # a2是字符串类型的 '2' 了



list2 = [35, 30, 91, 53]
# map(str, list2)是把list2里的元素都变成字符串了了，
# 但返回的是一个迭代器
# 好像在 API 里，iterable = iterator 的用法
# 最后 copy_2 是按字符串排序的, 但是按照字符串里的第2个元素排序的
copy_2 = sorted(map(str, list2), key = lambda c : c[1])
# API：sorted(iterable, *, key=None, reverse=False)

# 而这样只是把 list3变成一个迭代器
list3 = [2, 4, 1, 3]
myit = iter(list3)
# 注意，迭代器是不能像下面一行这样用的，会报错 'list_iterator' object has no attribute 'sort'
# myit.sort()
# 迭代器要sort，必须下面一行这样写：
# new_list直接是一个list = [1, 2, 3, 4]
new_list = sorted(myit)
# 或者像下面这样用
for each in sorted(myit):
  print(each)


mytuple = ("apple", "banana", "cherry")
# The for loop actually creates an iterator object
# and executes the next() method for each loop.
for x in mytuple:
  print(x)