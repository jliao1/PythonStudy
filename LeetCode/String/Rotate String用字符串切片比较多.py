'''
Description
Given a string of char array and an offset, rotate the string by offset in place. (from left to right).
In different languages, str will be given in different ways. For example, the string "abc" will be given in following ways:

Python：str = ['a', 'b', 'c']

注意理解题意，in place的意思是，就直接在原来的输入的 str 上改，
'''

# 切片
def rotateString(self, s, offset):
        # write your code here
        if not s:
            return None
        total_length = len(s)
        if offset > total_length:
            offset = offset % total_length

        index = total_length - offset
        a = s[:index]
        b = s[index:]
        s[:offset] = b
        s[offset:] = a


class Solution:
    # @param s: a list of char
    # @param offset: an integer
    # @return: nothing
    #我自己写的版本0，好像跟版本1没啥区别
    def rotateString0(self, s, offset):
        # write you code here
        if len(s) > 0:
            offset = offset % len(s)
        after = s[len(s)-offset:] + s[:len(s)-offset]

        for i in range(len(after)):
            s[i] = after[i]

    def rotateString1(self, s, offset):
        # write you code here
        if len(s) > 0:
            offset = offset % len(s)

        temp = (s + s)[len(s) - offset: 2 * len(s) - offset]

        for i in range(len(temp)):
            s[i] = temp[i]

    # Using python array implement method Array[::-1] do the reverse
    def rotateString2(self, str, offset):
        # write your code here
        if len(str) > 0:
            offset = offset % len(str)
        M = offset

        str[:-M] = str[:-M][::-1]

        str[-M:] = str[-M:][::-1]

        str[:] = str[::-1]


    def rotateString3(self, s, offset):
        # write your code here
        if not s:
            return None
        total_length = len(s)
        if offset > total_length:
            offset = offset % total_length
        # 整体反转
        self.reverse(s, 0, total_length - 1)
        # 反转左边
        self.reverse(s, 0, offset - 1)
        # 反转右边
        self.reverse(s, offset, total_length - 1)

    def reverse(self, s, left, right):
        while left <= right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

if __name__ == '__main__':
    sol = Solution()
    str = "abcdefg"
    str = list(str)

    res = sol.rotateString2(str,3)
    print(str)