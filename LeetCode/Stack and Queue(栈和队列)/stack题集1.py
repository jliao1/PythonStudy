import collections
class Solution:

    # Leetcode Easy 20. Valid Parentheses
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
    # Leet code Medium 394. Decode String  用recursion写法，其实更好理解
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
            stack.append( ''.join( reversed(str) ) * repeats )

            # str先翻转(变成原来的的排序)，再重复repeat次，然后再push进stack里
        return ''.join(stack)




import collections
if __name__ == '__main__':

    s ='abc'

    sol = Solution()
    res = sol.decodeString4("3[a2[pq]]de2[mn]")

    print(res)
