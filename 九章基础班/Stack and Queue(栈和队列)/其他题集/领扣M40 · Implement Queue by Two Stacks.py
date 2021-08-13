class Queue:
    def __init__(self, ):
        # do intialization if necessary
        self.stack1 = []
        self.stack2 = []

    def push(self, element):
        # write your code here
        self.stack1.append(element)

    def pop(self, ):
        self.top()
        return self.stack2.pop()

    def top(self, ):
        # 如果stack2为空，而stack1不为空，就把stack1里的元素搞到stack2里面来
        # 不然就等 stack2都pop空了，再把stack1搞过来
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        return self.stack2[-1]

if __name__ == '__main__':
    s = Queue()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    s.push(6)
    s.push(7)

    a = s.pop()
    b = s.pop()
    c = s.pop()
    s.push(10)
    s.push(20)
    d = s.pop()
    e = s.pop()
    pass
