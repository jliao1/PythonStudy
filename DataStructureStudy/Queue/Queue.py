from Linkedlist import ListNode

class Queue:
    """使用Linked list模拟队列"""
    def __init__(self):
        self.count = 0
        self.head = None
        self.tail = None   # 这是个特殊的单链表，要追踪尾部的，这样取尾部才能O(1)

    def enqueue(self, value):
        node = ListNode(value)
        if self.head is None:    # 如果头是空的，说明这个队列一个元素都没有
            self.head = node     # 那么head 和 tail 就都指向这个元素
            self.tail = node
        else:                      # 否则不是空的，说明这个队列有元素
            self.tail.next = node  # 那么node加在尾部
            self.tail = node       # 更新尾部
        self.count += 1

    # 出队列，从头部出
    def dequeue(self):
        if self.head is None:
            raise Exception('This is a empty queue')
        cur = self.head       # 先把头部存起来（因为后面要返回值）
        self.head = cur.next  # 队列的头部往后指一位   如果不返回删掉的头部值，其实只要 self.head = self.head.next
        self.count -= 1
        return cur.val        # 返回从头删掉的值

    def is_empty(self):
        return self.head is None  # 或者判断 self.count == 0

    def size(self):
        return self.count
