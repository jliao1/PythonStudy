from collections import deque

class Solution:

    # Leetcode Easy 20. Valid Parenthesis
    def isValid(self, s: str) -> bool:
        if not s:
            return True

        stack = []
        for c in s:
            # 压栈
            if c == '{' or c == '[' or c == '(':
                stack.append(c)
            else:
                # 栈需非空，不然后面stack[-1]会报 run time error
                if not stack:
                    return False

                if c == ']' and stack[-1] != '[' or c == ')' and stack[-1] != '(' or c == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()

        return not stack  # 最后stack空了就return True，不然就return False

    # Leet code Medium 71. Simplify Path
    def simplifyPath(self, path: str) -> str:
        """
        为什么想到用stack？
        因为有时向左回到上级目录，有时候向右去子目录
        像一个栈有时弹出元素（当前目录），有时候压入元素（子目录）

        时间复杂度：O(n): First, we spend O(N) trying to split the input path into components and then we process each component one by one which is again an O(N). 最后join那里其实也O(N)了
        空间复杂度：O(n): Actually, it's 至少 2N because we have the array that contains the split components and then we have the stack.
        """
        arr = path.split('/')  # 空间时间O(n)，会以/为分隔符，产生一个list
        stack = []             # 空间O(n)

        # Process
        for s in arr:          # 时间O(n)
            # case 1: 若栈不为空，返回上级菜单，如果为空，什么也不做
            if '..' == s:
                # 肌肉记忆：pop之前一定要看看是否为空
                if stack:
                    stack.pop()
                continue

            # case 2: 什么也不做
            if '' == s or '.' == s:
                continue

            # normal case: 一般的字母
            stack.append(s)

        # 到此，stack里的都是最精简有效了

        # 如果栈为空，直接返回 '/'
        if not stack:
            return '/'

        # 如果栈不为空，构造函数返回结果
        # 以 '/' 开头 (但注意不要以'/'结束)
        # 写法1：没开辟新空间了
        return '/' + '/'.join(stack)   # 时间O(n)

        # # 写法2：这个新开辟了 deque 空间，写法1没有新开空间
        # res = collections.deque()
        #
        # while stack:
        #     res.appendleft(stack.pop())
        #     res.appendleft('/')
        #
        # return ''.join(res)

        # # 写法3：
        # ans = ''
        # while stack:
        #     ans = '/' + stack.pop() + ans
        # return ans

    # Leet code Medium 394. Decode String  用了1个stack
    def decodeString1(self, s: str) -> str:
        """
        为什么想到用stack？
        这是一个嵌套展开过程。一般如果是嵌套的感觉，就往 queue 和 stack 方向去想想

        Time complexity:
        worst case下是 O(maxTimes^Nested·N)
            maxCnt 指的是 maximum value of times
            Nested 指的是 the count of nested times values
            N is the maximum length of sub encoded string
            比如 20[a10[bcc]15[de]]   maxTimes，Nested是2，N是3
        感觉空间复杂度也是 O(maxCnt^cntNested·N)

        小tip 1:
        如果要 顺序 弹出stack里的内容，可以用deque结构：
        strs = collections.deque()
        while stack and not isinstance(stack[-1], int):
            strs.appendleft(stack.pop())

        小tip 2:
        isdigit() 是对string用的，比如 ch 是个string 类型，就可以判断 if ch.isdigit()
        isinstance() 是对任意object用的，判断它是不是你期待的那个类型，比如 if not isinstance(stack[-1], int):
        """
        stack = []
        times = 0

        for ch in s:

            # case1: 如果是数字，继续 process cnt
            if ch.isdigit():
                # 字符 -> 数字 的一种通用写法
                times = times * 10 + int(ch)

            # case2: 如果是[, 说明 cnt 处理好了，把它压栈，然后 cnt 归零
            elif ch == '[':
                stack.append(times)
                times = 0

            # case3: 如果是], 把stack里的每个字符都弹出(用到队列的结构)，组成一个字符串
            #                cnt，把字符串压栈 cnt 次
            elif ch == ']':
                # 开始取字符串
                curr_str = collections.deque()
                #    如果栈非空    如果栈顶不是int (因为你期待的是字符，当碰到int时，while就结束了)
                while stack and not isinstance(stack[-1], int):
                    curr_str.appendleft(stack.pop())  # strs 必须要用双端队列deque结构，
                                                  # 因为 stack 顶 最先 pop 出来的，要放在 strs 最后
                # 开始取数字，走到这里，说明栈顶是 int 类型的 number 了，就pop它
                repeat_time = stack.pop()

                # 这两句，才是本code最expensive的操作，
                #       主要分析它的 time complexity
                #       和 stack 的 space complexity
                for _ in range(repeat_time):  # 这句时复是 O()
                    s = ''.join(curr_str)        # 这句是O(strs长度)
                    stack.append(s)

            # case4: 如果是字母，压栈
            elif ch.isalpha():  # 其实也可以写成 else: # 因为剩下的也只能是 字母 的情况了
                stack.append(ch)

        # connecting processed values in the stack
        return ''.join(stack)

    # Leet code Medium 394. Decode String  用了2个stack
    def decodeString2(self, s: str) -> str:
        """
        countStack: The stack would store all the integer k.
        stringStack: The stack would store all the decoded strings.
        python 这种写法我觉得跟方法1的时间空间复杂度是一样的

        【小tip】
        str1 = 'abcd'
        str2 = str * 5
        这两句的time complexity 应是总长度，字符串*按正常思路是会计算总空间然后再开空间填入，底层具体是怎么实现我确实不确定，面试的时候只要不是用for循环加基本上数量级是不会超过其他的，所以一般我们也不常考虑这部分的具体实现
        """
        numStack = []
        strStack = []

        num = 0
        currStr = []

        idx = 0
        while idx < len(s):
            # 找 number
            while idx < len(s) and s[idx].isdigit():
                num = num * 10 + int(s[idx])
                idx += 1

            # 找 字符
            while idx < len(s) and s[idx].isalpha():
                currStr.append(s[idx])
                idx += 1

            if idx < len(s) and s[idx] == '[':
                # 数字 压栈
                numStack.append(num)
                # 字符串 压栈
                strStack.append(''.join(currStr))
                # reset
                num = 0
                currStr = []
                # increment
                idx += 1

            if idx < len(s) and s[idx] == ']':
                #        栈顶的pop出来  +      curr的 * number次
                decodeString = strStack.pop() + ''.join(currStr) * numStack.pop()
                #       更新 curr
                currStr = []
                currStr.append(decodeString)
                idx += 1

        return ''.join(currStr)

    def __init__(self):
        self.idx = 0
    # Leet code Medium 394. Decode String  用recursion写法依次接到屁股上，其实更好理解
    def decodeString3(self, s: str) -> str:
        """
        Time complexity: worst case下依然是 O(maxTimes^Nested·N)
        感觉空间复杂度也是 O(maxCnt^cntNested·N)     但这里还要加个O(n),n是string的长度,这是栈空间
        """
        res = []
        while self.idx < len(s) and s[self.idx] != "]":
            if not s[self.idx].isdigit():
                res.append(s[self.idx])
                self.idx += 1
            else:
                num = 0
                # build n while next character is a digit
                while self.idx < len(s) and s[self.idx].isdigit():
                    num = num * 10 + int(s[self.idx])
                    self.idx += 1
                # ignore '['
                self.idx += 1
                decodedString = self.decodeString3(s)
                # ignore '['
                self.idx += 1
                while num > 0:
                    res.append(decodedString)
                    num -= 1
        return ''.join(res)

    # Leet code Medium 394. Decode String  这个思路：入栈，碰到结束标志后再弹栈，逆序处理，逻辑也很好理解
    def decodeString4(self, s: str) -> str:
        stack = []

        for c in s:

            if c != ']':       # ']' 才是结束标志
                stack.append(c)
                continue

            # 能到这里说明遇到']'了，开始逆序处理了
            str = [] # 字符部分

            while stack and stack[-1] != '[':
                str.append( stack.pop() )  # str 跟 input 的字符串比，是逆序排列的

            # 手动 pop '['
            stack.pop()

            # 处理数字（这个数字就是重复次数）
            # 注意pop是逆序的，所以要逆序处理字符，组合成最终的数字
            repeats = 0                 # 比如 '234' = 4 * 1 + 3 * 10^1 + 2 * 10^2
            base = 1
            while stack and stack[-1].isdigit():
                repeats += int( stack.pop() ) * base
                base *= 10
                                  # 为啥要 reversed() 因为之前的 str 被 pop 出时是逆序排列的
                                  # 所以处理完的str也是逆序的，就要先翻转(恢复正顺)，再重复repeat次，然后再push进stack里
                                  # 其实好像也可以 在最开始就整个翻转
            stack.append( ''.join( reversed(str) ) * repeats )


        return ''.join(stack)

    # Lintcode 978 Medium · Basic Calculator (operand都是正数) 自己累死累活 用双端队列 做的 recursive 版本，结果依次接在屁股上
    def calculate1(self, s):
        """
        时间空间复杂度，我分析不清楚了   这个好像也是O(n)吧，每个字符有可能处理2次，一次是入栈，第二次是出栈计算
        """
        numQ = deque()
        operQ = deque()

        res = 0
        while self.idx < len(s):
            # 处理数字
            if s[self.idx].isdigit():
                num = 0
                while self.idx < len(s) and s[self.idx].isdigit():
                    num = num * 10 + int(s[self.idx])
                    self.idx += 1
                # num 进数字队, 然后归零
                numQ.append(num)


            # 遇到空格，啥都不处理
            while self.idx < len(s) and s[self.idx] == ' ':
                self.idx += 1

            # + - 入符号队
            if self.idx < len(s) and s[self.idx] in '+-':
                operQ.append(s[self.idx])
                self.idx += 1

            # ( 就进入recursion
            if self.idx < len(s) and s[self.idx] == '(':
                self.idx += 1  # 跳过 ‘(’
                tempAnswer = self.calculate(s)
                numQ.append(tempAnswer)

            # 处理 数字和符号队，计算res，并return
            if self.idx < len(s) and s[self.idx] == ')':
                while operQ and numQ:
                    num1 = numQ.popleft()
                    num2 = numQ.popleft()
                    oper = operQ.popleft()

                    if oper == '+':
                        numQ.appendleft(num1 + num2)
                    elif oper == '-':
                        numQ.appendleft(num1 - num2)
                # 处理完了，更新 index
                self.idx += 1
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

    # Lintcode 978 Medium · Basic Calculator (operand都是正数) 先翻转，再入栈出栈，翻转就是逆序处理 其实这是力扣224第一个解法
    def calculate2(self, s: str) -> int:
        """
        时间 空间 复杂度都是 O(n)     n 是 the length of the string
        但 each character can potentially get processed twice,
        once when it's pushed onto the stack
        and once when it's popped for processing of the final result (or a subexpression)
        """
        stack = []
        n, operand = 0, 0

        for i in range(len(s) - 1, -1, -1):
            ch = s[i]

            if ch.isdigit():

                # Forming the operand - in reverse order.
                operand = (10**n * int(ch)) + operand
                n += 1

            elif ch != " ":
            # 到此时, 说明 ch既不是数字，也不是空格，那就开始处理 ) ( + - 这些符号了

                # 如果 n 不等于0，这条语句为 True
                # n不等于0时，说明此时是有 operand 的, 那就把它入栈
                if n:
                    # Save the operand on the stack as we encounter some non-digit.
                    stack.append(operand)
                    # 入栈后，n和operand归零
                    n, operand = 0, 0

                # When we encounter an opening parenthesis (, 由于我们一开始是逆序遍历的，so ( means an expression just ended.
                # This calls for evaluation of the current sub-expression
                if ch == '(':
                    res = self.evaluate_expr2(stack)

                    # 遇到 ( 开始计算，计算完了，意味着处理完了 (sub-expression), 就 pop 掉 ')' 符号
                    stack.pop()

                    # Append the evaluated result to the stack.
                    stack.append(res)

                # For other non-digits just push onto the stack.
                else:
                    stack.append(ch)

        # Push the last operand to stack, if any.
        if n:
            stack.append(operand)

        # Evaluate any left overs in the stack.
        return self.evaluate_expr2(stack)
    def evaluate_expr2(self, stack):
        """
        如何计算呢？
        by popping operands and operators off the stack till
        we pop corresponding closing parenthesis.
        """
        res = stack.pop() if stack else 0

        # Evaluate the expression till we get corresponding ')'  由于是逆序处理，)  成了sub-expression结束的标志符
        while stack and stack[-1] != ')':
            sign = stack.pop()
            if sign == '+':
                res += stack.pop()
            else:
                res -= stack.pop()
        return res

    # Leetcode 224 Hard · Basic Calculator (operand含正/负数) 找出一套规律，两两结合, operator滞后currNumber一轮处理
    def basicCaculator(self, s):
        """
        杂找出规律的呢？！思路：
        由于operand不只是正数，还有负数，
        所以 use - operator as the magnitude for the operand to the right of the - operator
        Once we start using - as a magnitude for the operands,
        we just have one operator left which is +, and + is associative 加号是可以前后随意结合关联的
        因此 Thus evaluating the expression from right or left, won't change the result
        Eg: A−B−C could be re-written as A+(−B)+(−C)

        这道题的做法是
        we will be evaluating most of the expression on-the-go
        + - ）mark the end of an operand，when encountered + - ), it's time to use currOperand and the sign to the left of it for evaluation

        we could add the values to the result beforehand and keep track of the lastest calculated number to the result,
        thus eliminating the unnecessary need for the stack 减少了栈使用，
        when encountered (，说明新的 sub-expression 开始了，这种时候是 necessary 的需要把之前的 result 先入栈

        遇到 ( 是入栈标志
        遇到 + - ）是计算标志
        遇到 ） 是 pop 整合的标志

        空间复杂度O(n)，n 是 the length of the string, stack 最多不可能装超过 O(n) 个元素
        时间复杂度O(n)，n 是 the length of the string.
                     这比领扣978 calculate2 好的点在于, every character in this approach will get processed exactly once
                                         而 领扣978 each character can potentially get processed twice,
                                                  once when it's pushed onto the stack
                                                  and once when it's popped for processing of the final result (or a subexpression).
                                                  That's why this approach is faster.
        """
        # 一个功能栈
        stack = []
        # Operand表示当前的操作数，初始状态下它是0
        currOperand = 0
        # sign表示当前的操作数的正负，初始状态下它是1
        currSign = 1  # 1是被加，-1是被减
        # 存结果的
        currRes = 0

        # 开始遍历啦
        for c in s:

            # case1：如果是数字，更新 current number
            if c in '1234567890':  # 这句也可以写成 if c.isdigit():
                currOperand = currOperand * 10 + int(c)

            # case2：遇到 +- 计算符号，就计算出当然结果，更新 sign来决定下轮是加还是减currNum
            #   英文解释是 when encountered + - ), it's time to use currOperand and the sign to the left of the operand for evaluation
            elif c in '+-':
                # Evaluate the expression to the left, save it
                currRes = currRes + currSign * currOperand
                currOperand = 0  # 得出暂时的 result 后 reset 一下 operand for next use

                # save the encountered signed for next time use
                if c == '+':
                    currSign = 1
                elif c == '-':
                    currSign = -1

            # 要把 ( 之前的状态 入栈保存了
            elif c == '(':
                # 入栈
                stack.append(currRes)
                stack.append(currSign)

                # Reset operand and result,
                # as if new evaluation begins for the new sub-expression
                currSign = 1  # new sub-expression 中 currSign 初始状态是 1
                currRes = 0  # new sub-expression 中 currRes 初始状态是 0

            # 要计算从此 ) 到上一个 ( 的 currRes 了
            elif c == ')':
                # when encountered + - ), it's time to use currOperand and the
                # sign to the left of the operand for evaluation，save it
                currRes = currRes + currSign * currOperand
                currOperand = 0  # 得出暂时的 result 后 reset 一下 operand for next use

                # and ')' also marks end of expression within a set of parenthesis
                # Its result is multiplied with sign on top of stack as stack.pop() is the sign before the parenthesis
                lastSign = stack.pop()
                # Then add to the next operand on the top as stack.pop() is the result calculated before this parenthesis
                lastRes = stack.pop()
                # update currRes = (lastRes on stack) + (lastSign on stack * (currRes from parenthesis))
                currRes = lastRes + lastSign * currRes  # stack[-1] pop出来的是 ( 前入栈的 sign

            # 其实如果以上if都没执行，就是遇到空格了，不过空格情况下，我们什么也不用做

        # 最后一次计算结果了，并返回 (如果这已经是最后一个数了，不需要计算了，那么此时的currOperand是0，相当于就只返回currRes啦)
        return currRes + currSign * currOperand

    # leetcode 227 Medium Basic Calculator II ()，本题不包含括号和非负数, 写法不错用到了enumerate
    def basicCalculatorII1(self, s: str) -> int:
        """
        思路跟力扣224像的地方在于，operator 滞后 currNumber 一轮 进行处理
        不一样的地方在于，加工好每个数再入栈，最后sum(stack)算出结果
        因为The expressions只需要are evaluated from left to right and the order of evaluation depends on the Operator Precedence
        (但这种写法不能处理非负数啦，比如处理不了这种case: +48 + 2*-48)

        时间空间复杂度O(n)，n是s的长度，
        但potentially有可能处理一个字符2次(乘除时出栈再入栈)，处理每一个值也是2次(入栈再最后计算)
        版本2更优化一些
        """
        stack = []
        currNum = 0
        lastOperator = '+' # 初始值

        for i, char in enumerate(s):
            # case1：c是数字，处理下
            if char.isdigit():
                currNum = 10 * currNum + int(char)

            # case2：如果遇到的，不是数字也不是空格(那就是那些 加减乘除符号了)
            #        或 i 已经扫到 最后一位，哪怕char是digit，也要处理
            #        这时候开始处理
            if (not char.isdigit() and not char.isspace()) or i == len(s) - 1:
                # case2.1 遇到 + - 时，连上 operator proceeding it 再入栈
                if lastOperator == '+':
                    stack.append(+ currNum)
                elif lastOperator == '-':
                    stack.append(- currNum)

                # case2.2 遇到 * / 时 弹栈顶数字，与currNum计算后(因为*/优先级高)，把结果入栈
                elif lastOperator == '*':
                    stack.append(stack.pop() * currNum)
                elif lastOperator == '/':
                    # (-3//4) 等于 -1, 所以要用 int(-3/4) 才能等于0
                    stack.append(int(stack.pop() / currNum))

                # 把 lastOperator 更新成本轮遇到测char
                lastOperator = char
                # 处理完这一轮了，currNum要清零
                currNum = 0

        return sum(stack)

    # leetcode 227 Medium Basic Calculator II ()，本题不包含括号和非负数
    def basicCalculatorII2(self, s: str) -> int:
        """
        思路跟 力扣224相同的地方在于，
        we add the values to the result beforehand and keep track of the last calculated number,
        thus eliminating the need for the stack
        但在计算 3-2时，是把搞成 3 + -2 的形式了，所以

        时间 O(n)  而且不用每个字符都处理2次，这只需要处理1次，所以更快
        不用栈了，所以空间O(1)
        """
        if not s:
            return 0

        currNum, lastNum, res = 0,0,0

        lastOperator = '+'

        for i, char in enumerate(s):
            if char.isdigit():
                currNum = 10 * currNum + int(char)

            if (not char.isdigit() and not char.isspace()) or i == len(s)-1:
                if lastOperator in '+-':
                    res = res + lastNum
                    # 这一步是 把减号搞成 + 一个负数的形式
                    lastNum = currNum if lastOperator == '+' else -currNum
                elif lastOperator in '*/':
                    lastNum = lastNum * currNum if lastOperator == '*' else int(lastNum / currNum)

                lastOperator = char
                currNum = 0

        return res + lastNum






import collections
if __name__ == '__main__':
    q = collections.deque()
    q.append(1)
    q.append(2)
    q.append(3)
    print(q)
    print(q.pop())
    print(q)

    # sol = Solution()
    # # res = sol.basicCalculatorII("+48 + 2*-48")
    # res = sol.basicCalculatorII2("14-3/2")
    # print(res)
    # list = [1,2,3,4,5]
    # print(min(list))


