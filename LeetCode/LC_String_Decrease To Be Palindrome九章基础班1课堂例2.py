# leetCode：Decrease To Be Palindrome
# 其实没懂题目: 老师讲后懂了，题目提取的关键是 允许把当前字母变成前面一个字母，最少需要几步操作，来变成回文串


'''
知识点：
（1）ord() 是返回对 某单个字符的 应的ASCII 数值

（2）回文串：
  1。如何判断？用两个指针向中间走（或者从中间向两边走），找双胞胎
  2。性质：
    1. 对称性，知道左边一半，就知道右边一半（做个镜像就是右边一半，合起来就是个回文串）
    2. (重要!!!) 奇偶性：最多有一个字母出现奇数次，其他字母都出现偶数次。
    3. 回文串的长度有可能是奇数(轴心是中间的数，这个数出现了奇数次)，有可能是偶数(轴心是中间的缝隙)
'''


class Solution:
    """
    @param s: the string s
    @return: the number of operations at least
    """

    # # 写法一
    # def numberOfOperations(self, s):
    #     # 先处理edge case
    #     if s is None:
    #         return 0
    #
    #     minStep = 0
    #     # 需要两个指针 (小技巧，双指针)
    #     left, right = 0, len(s) - 1
    #
    #     while left < right:
    #         # 'a' 'c'
    #         minStep += abs(ord(s[left]) - ord(s[right]))
    #
    #         left += 1  # 左指针向右移动
    #         right -= 1  # 右指针向左移
    #
    #     # 然后等左右指针相碰，就结束了
    #     return minStep

    # 写法二: 九章老师上课讲的
    def numberOfOperations(self, s):
        n = len(s)
        res = 0
        # 遍历字符的左半边就好（因为 汇文串特性就是左右对撑的）
        for i in range(n // 2):                        # range(startIndex, endIndex, step)
            # 利用对称性，从两边往中间 遍历
            res += abs(ord(s[i]) - ord(s[n - 1 - i]))  # range()内如果只有一个参数, 比如这里是n//2, 说明 startIndex默认是0, endIndex是n//2, step默认是1
                                                       # n//2的意思是向下取整
        return res


