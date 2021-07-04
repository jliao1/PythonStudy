
class Solution:
    def __init__(self):
        self.i = 0

    # Lintcode 978 Medium · Basic Calculator (operand都是正数) 自己累死累活 用2个stack 做的 recursive 版本
    def calculate(self, s):
        """
        时间空间复杂度，我分析不清楚了
        """
        numQ = deque()
        operQ = deque()

        res = 0
        while self.i < len(s):
            # 处理数字
            if s[self.i].isdigit():
                num = 0
                while self.i < len(s) and s[self.i].isdigit():
                    num = num * 10 + int(s[self.i])
                    self.i += 1
                # num 进数字队, 然后归零
                numQ.append(num)


            # 遇到空格，啥都不处理
            while self.i < len(s) and s[self.i] == ' ':
                self.i += 1

            # + - 入符号队
            if self.i < len(s) and s[self.i] in '+-':
                operQ.append(s[self.i])
                self.i += 1

            # ( 就进入recursion
            if self.i < len(s) and s[self.i] == '(':
                self.i += 1  # 跳过 ‘(’
                tempAnswer = self.calculate(s)
                numQ.append(tempAnswer)

            # 处理 数字和符号队，计算res，并return
            if self.i < len(s) and s[self.i] == ')':
                while operQ and numQ:
                    num1 = numQ.popleft()
                    num2 = numQ.popleft()
                    oper = operQ.popleft()

                    if oper == '+':
                        numQ.appendleft(num1 + num2)
                    elif oper == '-':
                        numQ.appendleft(num1 - num2)
                # 处理完了，更新 index
                self.i += 1
                if numQ:
                    return numQ.popleft()

        while operQ and numQ:
            num1 = numQ.popleft()
            num2 = numQ.popleft()
            oper = operQ.popleft()

            if oper == '+':
                numQ.appendleft(num1 + num2)
            elif oper == '-':
                numQ.appendleft(num1 - num2)

        if numQ:
            return numQ.popleft()

    # Leetcode 224 Hard.Basic Calculator (operand含正/负数) 假想一套full的规律
    def basicCalculator(self, s):
        """
        由于operand含正/负数，所以 operand前的sign总要跟operand紧密结合在一起。那么就两两结合，才不会出幺蛾子
        遇到 ( 是入栈标志
        遇到 + - ）是计算标志
        遇到 ） 是 pop 整合的标志

        """
        # 一个功能栈
        stack = []
        # number表示当前的操作数
        currNum = 0
        # sign表示当前的操作数的正负，应该被加还是被减
        currSign = 1  # 1是被加，-1是被减
        # 存结果的
        currRes = 0

        # 开始遍历啦
        for c in s:

            # case1：如果是数字，更新 current number
            if c in '1234567890':  # 这句也可以写成 if c.isdigit():
                currNum = currNum * 10 + int(c)

            # case2：遇到 +- 计算符号，就计算出当然结果，更新 sign来决定下轮是加还是减currNum
            elif c in '+-':
                # 遇到符号就要先计算一下 currRes
                currRes += currSign * currNum

                # 得出暂时的 result 后
                # 要 reset 一下 current number
                currNum = 0

                # 然后 reset 一下 currSign 给下一轮使用
                # 若 + 那么 currSign 是 +1，说明下轮计算会 + currNum
                # 若 - 那么 currSign 是 -1，说明下轮计算会 - currNum
                if c == '+':
                    currSign = 1  # 这会导致
                elif c == '-':
                    currSign = -1

            # 要把 ( 之前的状态 入栈保存了
            elif c == '(':
                # 入栈
                stack.append(currRes)
                stack.append(currSign)

                # reset current Sign 和 current result
                currSign = 1
                currRes = 0  # 更新currRes，让currRes去保存 ( 后的新一轮儿的东西了

            # 要计算从此 ) 到上一个 ( 的 currRes 了
            elif c == ')':
                # 计算出 (……) 内的 current result 然后 current number 更新成0
                currRes += currSign * currNum
                currNum = 0

                currRes *= stack[-1]  # stack[-1] pop出来的是 ( 前入栈的 sign
                currRes += stack[-2]  # stack[-2] pop出来的是 ( 前入栈的 result 跟 currRes 相加，结果更新成现场的 currRes

                # 为啥要 stack[:-2] 这样呢？ 这样是直接pop了最后两个。前两句取了它们的值，现在pop它们
                stack = stack[:-2]

            # 其实如果以上if都没执行，就是遇到空格了，不过空格情况下，我们什么也不用做

        # 计算结果，最后一次了
        currRes += currSign * currNum
        return currRes

import queue
from collections import deque

if __name__ == '__main__':
    s = "1 + (2+ 3)"
    sol = Solution()
    val = sol.basicCalculator(s)
    print(val)
