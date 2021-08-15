'''
leetCode：Decrease To Be Palindrome
其实没懂题目: 老师讲后懂了，题目提取的关键是 允许把当前字母变成前面一个字母，最少需要几步操作，来变成回文串

Description
Given a string s with a-z. We want to change s into a palindrome with following operations:

1. change 'z' to 'y';
2. change 'y' to 'x';
3. change 'x' to 'w';
................
24. change 'c' to 'b';
25. change 'b' to 'a';
Returns the number of operations needed to change s to a palindrome at least.
'''



class Solution:
    """
    @param s: the string s
    @return: the number of operations at least
    """

    # 写法一
    def numberOfOperations1(self, s):
        # 先处理edge case
        if s is None:
            return 0

        minStep = 0
        # 需要两个指针 (小技巧，wk1_双指针)
        left, right = 0, len(s) - 1

        while left < right:
            # 'a' 'c'
            minStep += abs(ord(s[left]) - ord(s[right]))

            left += 1  # 左指针向右移动
            right -= 1  # 右指针向左移

        # 然后等左右指针相碰，就结束了
        return minStep

    # 写法二: 九章老师上课讲的
    def numberOfOperations2(self, s):
        n = len(s)
        res = 0
        # 遍历字符的左半边就好（因为 汇文串特性就是左右对撑的）
        for i in range(n // 2):
            print( i)# range(startIndex, endIndex, step)
            # 利用对称性，从两边往中间 遍历
            res += abs(ord(s[i]) - ord(s[n - 1 - i]))  # range()内如果只有一个参数, 比如这里是n//2, 说明 startIndex默认是0, endIndex是n//2, step默认是1
                                                       # n//2的意思是向下取整
        return res

    # 写法3:  我自己写的
    def numberOfOperations3(self, s):
        cnt = 0

        if s is None or s == '':
            return cnt

        n = len(s)

        mid = n // 2

        for i in range(0, mid):
            if ord(s[i]) != ord(s[n - 1 - i]):
                cnt = cnt + abs(ord(s[i]) - ord(s[n - 1 - i]))

        return 1

if __name__ == '__main__':
    sol = Solution()
    res = sol.numberOfOperations3("abcd")
    s = '656667'
    len = len(s)  # 长度是6
    for i in range(len-1,-1,-1): # 从5依次递减2，直到0(不包括0)
        print(i)

