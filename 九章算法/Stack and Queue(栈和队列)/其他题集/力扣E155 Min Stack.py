
"""
这个思路 清奇！
一个栈，但每个元素是一个元组
元组的第一个值是 压入的数字
第二个值记录的是 当前最小值

时间复杂度是O(1)
时间复杂度是O(2n)，有可能很多元组的第二个值都是一样的，
就会存很多个相同的值，可以优化下，看版本2
"""
class MinStack1(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type val: int
        :rtype: None
        """
        if not self.stack:
            self.stack.append((x, x))
        else:
            currMin = self.stack[-1][1]
            self.stack.append((x, min(x, currMin)))

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1][0]

    def getMin(self):
        """
        :rtype: int
        """
        return self.stack[-1][1]


"""
这个方法使用了2个栈，除了本身的stack
push时只有元素更小的时候才放入这个栈
而pop时只有栈顶与stack栈顶相同时才弹出
方法1的空间其实是O(2n)，方法2的空间复杂度节省到了O(n)了
时间复杂度还是O(1)

但这也有个缺点 the same number is pushed repeatedly onto MinStack, 
and that number also happens to be the current minimum, there'll 
be a lot of needless repetition on the min-tracker Stack.
可以看下方法3，但个人认为方法3的角度有点刁钻了
"""
class MinStack2(object):

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        #                 这个小于等于号不能去掉，不然pop时候会出错
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.min_stack[-1] == self.stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]

"""
比方法2更省空间，虽然也还是在O(n)的纬度
不过这个思路真是有点刁钻了
"""
class MinStack3(object):
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        # We always put the number onto the main stack.
        self.stack.append(x)

        # If the min stack is empty, or this number is smaller than
        # the top of the min stack, put it on with a count of 1.
        if not self.min_stack or x < self.min_stack[-1][0]:
            self.min_stack.append([x, 1])

        # Else if this number is equal to what's currently at the top
        # of the min stack, then increment the count at the top by 1.
        elif x == self.min_stack[-1][0]:
            self.min_stack[-1][1] += 1

    def pop(self):
        # If the top of min stack is the same as the top of stack
        # then we need to decrement the count at the top by 1.
        if self.min_stack[-1][0] == self.stack[-1]:
            self.min_stack[-1][1] -= 1

        # If the count at the top of min stack is now 0, then remove
        # that value as we're done with it.
        if self.min_stack[-1][1] == 0:
            self.min_stack.pop()

        # And like before, pop the top of the main stack.
        self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1][0]


if __name__ == '__main__':
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(prerequisites[0][1])
# # Your MinStack object will be instantiated and called as such:
#     obj = MinStack2()
#     obj.push(1)
#     obj.pop()
#     param_3 = obj.top()
#     param_4 = obj.getMin()
#


