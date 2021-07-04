class Dequeue(object):

    def __init__(self):
        # do some intialize if necessary
        self.head, self.tail = None, None

    # @param {int} item an integer
    # @return nothing
    def push_front(self, item):
        # Write yout code here
        if self.head is None:
            self.head = Node(item)
            self.tail = self.head
        else:
            tmp = Node(item)
            self.head.prev = tmp
            tmp.next = self.head
            self.head = tmp

    # @param {int} item an integer
    # @return nothing
    def push_back(self, item):
        # Write yout code here
        if self.tail is None:
            self.head = Node(item)
            self.tail = self.head
        else:
            tmp = Node(item)
            self.tail.next = tmp
            tmp.prev = self.tail
            self.tail = tmp

    # @return an integer
    def pop_front(self):
        # Write your code here
        if self.head is not None:
            item = self.head.val
            self.head = self.head.next
            if self.head is not None:
                self.head.prev = None
            else:
                self.tail = None
            return item

        return -12

    # @return an integer
    def pop_back(self):
        # Write your code here
        if self.tail is not None:
            item = self.tail.val
            self.tail = self.tail.prev
            if self.tail is not None:
                self.tail.next = None
            else:
                self.head = None
            return item

        return -12

class Node():

    def __init__(self, _val):
        self.next = self.prev = None
        self.val = _val