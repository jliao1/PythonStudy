# leetCode：Decrease To Be Palindrome
# 其实没懂题目


class Solution:
    """
    @param s: the string s
    @return: the number of operations at least
    """

    def numberOfOperations(self, s):
        # 先处理edge case
        if s is None:
            return 0

        minStep = 0
        # 需要两个指针 (小技巧，双指针)
        left, right = 0, len(s) - 1

        while left < right:
            # 'a' 'c'
            minStep += abs(ord(s[left]) - ord(s[right]))

            left += 1  # 左指针向右移动
            right -= 1  # 右指针向左移

        # 然后等左右指针相碰，就结束了
        return minStep


'''
知识点：
ord() 是返回对 某单个字符的 应的ASCII 数值
'''