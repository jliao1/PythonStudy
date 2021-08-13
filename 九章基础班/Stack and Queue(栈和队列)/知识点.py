'''
【实现queue】
一般可以用这3种结构
list               # 一般不用这个，因为 从头部remove元素会消耗O(n)
collections.deque  # 推荐用这个，因为 python 底层实现起来快
queue.Queue        # 虽然时间复杂度跟 deque 一样，但是不推荐，因为底层实现慢

【实现stack】
一般用list来实现，推荐这个，因为这个list是个很通用的结构
也可以用 deque来实现

【双端队列】
Deque (Doubly Ended Queue) in Python is implemented using the module “collections“.
Deque is preferred over list in the cases where we need quicker append and pop operations from both the ends of container
# 导入方法是：
from collections import deque
# Declaring deque
queue = deque(['name','age','DOB'])
# 打印
print(queue)

【python的iterator没有hasnext怎么办？】
Use next(iterable, default_value) to get the next value in iterable, with the default_value as the value to be returned as the value to be returned if there is no next value.
print(next(iterator, None))
参考网页 https://www.kite.com/python/answers/how-to-use-hasnext-with-iterators-in-python

'''

