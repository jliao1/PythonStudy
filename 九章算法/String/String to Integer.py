'''
Description
Given a string, convert it to an integer. You may assume the string is a
valid integer number that can be presented by a signed 32bit integer (-2^31 ~ 2^31-1).

简单的做法就是return int(str)

令狐冲版本的是：
要考虑给的字符串是否有负号。
然后从高位开始循环累加。
转换公示如下
字符串：S_1S_2S_3S_4
'''



class Solution:
    # @param {string} str a string
    # @return {int} an integer
    def stringToInteger(self, str):
        # Write your code here
        num, sig = 0, 1

        # 处理负号
        if str[0] == '-':
            sig = -1
            str = str[1:] # 丢掉负号

        # 把字符串，一位一位地，转成数字
        for c in str:
            num = num * 10 + ord(c) - ord('0')

        return num * sig

if __name__ == '__main__':
    sol = Solution()
    res = sol.stringToInteger("-123")

    list = [1,2,3]
    print(list[1])