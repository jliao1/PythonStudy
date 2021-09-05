class ListNode:
    def __init__(self, val):
        self.val = val;
        self.next = None


class MyQueue:

    def __init__(self):
        self.head = None
        self.tail = None

    def enqueue(self, item):
        if not self.head:
            self.head = ListNode(item)
            self.tail = self.head

        self.tail.next = ListNode(item)
        self.tail = self.tail.next

    def dequeue(self):
        if not self.head:
            return -1

        res = self.head.val
        self.head = self.head.next

        return res

if __name__ == '__main__':
    sol = MyQueue()
    sol.enqueue(1)
    sol.enqueue(2)
    sol.enqueue(3)
    sol.dequeue()
    sol.enqueue(4)
    sol.dequeue()