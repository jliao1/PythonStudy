class InterfaceQueue:
    def push(self, element):
        pass

    # define an interface for pop method
    # write your code here
    def pop(self):
        pass

    # define an interface for top method
    # write your code here
    def top(self):
        pass

class LinkedListNode:
    def __init__(self, val):
        self.val = val
        self.pre = None
        self.next = None

# lintcode Easy 546 · Implement Queue by Interface
class MyQueue(InterfaceQueue):
    # do initialization if necessary. you can declare your private attributes here
    def __init__(self):
        self.before_head = LinkedListNode(-1)
        self.tail = self.before_head

    # implement the push method
    def push(self, val):
        self.tail.next = LinkedListNode(val)
        self.tail = self.tail.next

    # implement the pop method
    def pop(self):
        # case1: 如果是 空的
        if not self.before_head.next:
            return -1

        # 如果只有1个元素
        if self.tail == self.before_head.next:
            temp = self.before_head.next.val
            self.before_head.next = None
            self.tail = self.before_head
            return temp

        # normal case 有大于等于2个的元素
        temp = self.before_head.next.val
        self.before_head.next = self.before_head.next.next
        return temp



    # implement the top method
    # write your code here
    def top(self):
        # case1: 如果是 空的
        if not self.before_head.next:
            return -1

        # case2：若有大于等于1个元素
        temp = self.before_head.next.val
        return temp




if __name__ == '__main__':
    # Your MyQueue object will be instantiated and called as such:
    queue = MyQueue()
    queue.push(123)
    print( queue.top() ) # will return 123;
    print( queue.pop() )# will return 123 and pop the first element in the queue

