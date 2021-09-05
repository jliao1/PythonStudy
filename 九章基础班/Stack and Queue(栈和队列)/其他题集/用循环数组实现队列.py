class MyQueue1(object):
    """
    从我们之前的判断来看, tail == MAXSIZE ，当前队列已经满了，不能继续添加元素了，
    但是实际上我们还可以继 续添加元素。因此在使用数组实现队列时，可能会出现空间未有效用的情况，因此，我们有两种解决方法:
    1. 使用链表实现队列 看领扣492题
    2. 使用数组来实现循环队列  看解法2

    """
    def __init__(self):
        self.MAXSIZE = 4
        self.queue = [0] * self.MAXSIZE  # 占用空间复杂度
        # 主要是通过调节这两个双指针
        self.head = 0
        self.tail = 0


    def enqueue(self, item):
        queue = self.queue

        # 队列满
        if self.tail == self.MAXSIZE:
            return

        queue[self.tail] = item
        self.tail += 1


    def dequeue(self):
        queue = self.queue

        # 队列为空
        if self.head == self.tail:
            return -1

        item = queue[self.head]
        self.head += 1
        return item

# 数组实现循环队列：  循环效果用 取模运算实现
class MyQueue2(object):

    def __init__(self):
        self.SIZE = 1000
        self.queue = [0] * self.SIZE
        self.head = 0
        self.tail = 0

    def enqueue(self, item):
        queue = self.queue

        # 队列满
        if (self.tail + 1) % self.SIZE == self.head:
            return

        queue[self.tail] == item

        self.tail = (self.tail + 1) % self.SIZE


    def dequeue(self):
        # 若为空
        if self.head == self.tail:
            return -1

        item = self.queue[self.head]

        self.head = (self.head + 1) % self.SIZE

        return item