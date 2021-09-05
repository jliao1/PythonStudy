class Node:
    def __init__(self, val):
        self.val = val;
        self.next = None

# LintCode Easy 492 · Implement Queue by Linked List
class MyQueue1:

    def __init__(self):
        # before_head的next其实是head，目的是不用区分头节点和后面的点（不用分开操作）
        self.before_head = Node(-1)
        self.tail = self.before_head

    def enqueue(self, item):
        """
        @param: item: An integer
        @return: nothing
        """
        # make the input item as tail
        self.tail.next = Node(item)
        self.tail = self.tail.next

    def dequeue(self):
        """
        @return: An integer
        """
        # case1: check before_head.next不是空（要确保目前维护的这个queue不空，才能dequeue）
        if self.before_head.next is None:
            return -1

        # case2: 如果目前只有1个元素了
        if self.before_head.next == self.tail:
            # update tail
            self.tail = self.before_head

        # 取 head 的 value
        res = self.before_head.next.val

        # 然后把 head 给dequeue了（其实就是把before_head指针后移）
        self.before_head.next = self.before_head.next.next

        return res

# LintCode Easy 492 · Implement Queue by Linked List 我自己写的
class MyQueue2:

    def __init__(self):
        # 初始化一下，这里不要 self.tail = self.head 因为 self.head 此时是None接不上的
        self.head = None
        self.tail = None

    def enqueue(self, item):
        if not self.head:
            self.head = Node(item)
            self.tail = self.head
        else:
            self.tail.next = Node(item)
            self.tail = self.tail.next

    def dequeue(self):
        if not self.head:
            return -1

        res = self.head.val
        self.head = self.head.next

        return res