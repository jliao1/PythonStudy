'''
Description
Given a sentence of English, update the first letter of each word to uppercase.

The given sentence may not be a grammatical sentence.
The length of the sentence does not exceed 100.
'''



class Solution:
    """
    @param s: a string
    @return: a string after capitalizes the first letter
    """
    def capitalizesFirst(self, s):
        # Write your code here
        n = len(s)
        s1 = list(s)
        # 这个字符是整个字符串的第一个字母且不是空格，那么就把这个字符大写
        if s1[0] >= 'a' and s1[0] <= 'z':
            s1[0] = chr(ord(s1[0]) - 32)
        # 如果一个非空格字符前面有空格
        for i in range(1, n):
            if s1[i - 1] == ' ' and s1[i] != ' ':
                s1[i] = chr(ord(s1[i]) - 32)

        return ''.join(s1)