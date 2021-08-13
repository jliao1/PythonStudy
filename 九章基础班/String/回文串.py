'''

Description
Given a string, determine if it is a palindrome,
considering only alphanumeric字母文字的 characters and ignoring cases.

'''
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    #这是我自己写的版本
    def isPalindrome1(self, s):
        if s is None:
            return False
        if s == '':
            return True

        # 取出是 数字/字母 的字符，但这比令狐老师的，占用空间了
        slist = []
        for char in s:
            if ord(char) >= ord('a') and ord(char) <= ord('z'):
                char = char.upper()
            if ord(char) >= ord('0') and ord(char) <= ord('9') or ord(char) >= ord('A') and ord(char) <= ord('Z'):
                slist.append(char)

        left = 0
        right = len(slist) - 1

        while left < right:
            if slist[left] != slist[right]:
                return False
            left += 1
            right -= 1

        return True


    # lintcode Medium 415 · Valid Palindrome 令狐老师的版本
    def isPalindrome2(self, s):
        """
        相同的逻辑的代码，提成一个函数 is_valid_char() 避免代码耦合
        如果 and 和 or 在一个判断语句里太多太长了，也可以提成函数，增加可读性
        一般一个条件，有一个 and 或 or 最好，太多了降低 readability
        """
        if not s:
            return True

        left = 0
        right = len(s) - 1

        while left <= right:
            while left <= right and not self.is_valid_char(s[left]):
                left += 1

            while left <= right and not self.is_valid_char(s[right]):
                right -= 1
            #                          就算 s[left] 是数字也可以 s[left].lower() 不会报错啦
            if left <= right and s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1

        return True
    def is_valid_char(self, char):
        return char.isdigit() or char.isalpha()

    def validPalindrome(self, s):
        def is_odd(value):
            return not value % 2 == 0

        if not s:
            return True

        import collections

        dic = collections.Counter(s)

        odd_count = 0

        for v in dic.values():
            if is_odd(v):
                odd_count += 1

        return odd_count <= 2

    # lintcode Medium 891 · Valid Palindrome II
    def validPalindrome(self, s):
        if not s:
            return True

        left, right = self.find_difference(s, 0, len(s) - 1)

        if left < right:
            return self.is_palindrome(s, left + 1, right) | self.is_palindrome(s, left, right - 1)
        else:
            return True
    def is_palindrome(self, s, left, right):
        l, r = self.find_difference(s, left, right)
        if l < r:
            return False
        else:
            return True
    # 相同的功能的代码，最好提炼成子函数
    def find_difference(self, s, left, right):
        while left <= right and s[left] == s[right]:
            left += 1
            right -= 1

        return left, right

def is_odd(value):
    return not value / 2 == 0
if __name__ == '__main__':
    a = is_odd(99)

    sol = Solution()
    res = sol.validPalindromeTwo("abceca")
    print(res)

