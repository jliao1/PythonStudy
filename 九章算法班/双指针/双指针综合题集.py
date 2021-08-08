class Solution:

    # lintcode Medium 415 · Valid Palindrome 令狐老师的版本
    def isPalindrome(self, s):
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

    # lintcode Medium 891 · Valid Palindrome II 功能相同的代码，提炼成子函数
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

if __name__ == '__main__':

    sol = Solution()
    l = sol.twoSum1([1,0,-1], 0)