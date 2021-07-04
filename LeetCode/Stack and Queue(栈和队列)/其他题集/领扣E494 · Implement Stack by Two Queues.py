# lintcode E 494 · Implement Stack by Two Queues
from collections import deque

class Stack:
    def __init__(self):
        self.q1 = deque()  # 主
        self.q2 = deque()  # 辅

    def push(self, x):
        self.q1.append(x)

    def pop(self):
        # 把 q1 除了最后一个，其他所有元素都 搞到q2去
        for _ in range(len(self.q1) - 1):
            val = self.q1.popleft()
            self.q2.append(val)

        # q1 里的最后一个元素，就是我们要找的，就是我们要找的，取出来存在 val
        val = self.q1.popleft()

        # 交换 q1 和 q2
        self.q1, self.q2 = self.q2, self.q1

        # 返回 val
        return val

    def top(self):
        # pop 出来取值，再push回去
        val = self.pop()
        self.push(val)
        return val

    def isEmpty(self):
        return not self.q1


if __name__ == '__main__':
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    s.push(6)
    s.push(7)

    a = s.pop()
    b = s.pop()
    pass
