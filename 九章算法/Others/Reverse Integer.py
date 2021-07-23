'''
Description
Reverse digits of an integer.
Returns 0 when the reversed integer overflows 32-bit integer. 意思是范围在 (-2^31) ~ (2^32-1)

对于32位系统来说
unsigned的范围是 0 - 2^32
signed的范围是 (-2^31) ~ (2^32-1)
'''

class Solution2:
    """
    @param n: the integer to be reversed
    @return: the reversed integer
    """

    def reverseInteger(self, n):

        # single digit cases
        if abs(n) < 10: return n

        # reverse w/o neg sign
        # 字符串切片来翻转整数！！牛逼了！
        r = int(str(abs(n))[::-1])

        # overflow edge cases
        if r >= pow(2, 31):
            return 0
        # dont forget the neg sign
        return r if n > 0 else -r

# solution2： 令狐冲版本，但不太对，自己再改下
class Solution1:
    # @param {int} n the integer to be reversed
    # @return {int} the reversed integer
    def reverseInteger(self, n):
        if n == 0:
            return 0

        # 处理负号
        neg = 1
        if n < 0:
            neg, n = -1, -n

        # 翻转整数
        reverse = 0
        while n > 0:
            reverse = reverse * 10 + n % 10
            n = n // 10

        reverse = reverse * neg
                       # 1 << 31 就是2的31次方的意思
        if reverse < -(1 << 31) or reverse > (1 << 31) - 1:
            return 0
        return reverse


if __name__ == '__main__':
    sol = Solution2()
    res = sol.reverseInteger(-123)
    print( 'example'.strip('el') )

