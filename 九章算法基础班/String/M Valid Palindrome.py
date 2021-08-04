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
    def isPalindrome(self, s):
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


    # 令狐老师的版本
    def isPalindrome2(self, s):
        start, end = 0, len(s) - 1
        while start < end:
            # 如果start指向的不是字母，不是数字，就移动start指针
            while start < end and not s[start].isalpha() and not s[start].isdigit():
                start += 1
            # 如果end指向的不是字母，不是数字，就移动end指针
            while start < end and not s[end].isalpha() and not s[end].isdigit():
                end -= 1
            # 好了现在 start 和 end 都是字母或数字了
            if start < end and s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        return True


if __name__ == '__main__':
    sol = Solution()
    res = sol.isPalindrome('A man, a plan, a canal: Panama')
    print(res)

