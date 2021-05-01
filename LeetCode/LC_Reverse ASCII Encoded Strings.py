# Leetcode：Reverse ASCII Encoded Strings
# 思路：每两个数字是一对地翻译，比如 字符 ‘6’‘5’是一对 -> 65 -> 'A'   最后切片翻转字符串  或者一开始就可以倒序翻译


class Solution:
    """
    @param encodeString: an encode string
    @return: a reversed decoded string
    """

    def reverseAsciiEncodedString(self, encodeString):

        # 写法1：每两个数字是一对地翻译，比如 字符 ‘6’‘5’是一对 -> 65 -> 'A'   最后切片翻转字符串
        if encodeString is None:
            return ""

        res = ""  # 用来存储

        for i in range(0, len(encodeString), 2):  # 每两个两个数字地取
            # 比如 字符 ‘6’ ‘5’ 是一对 -> 65 -> 'A'
            # 字符 强制转换 int
            ascNumber = int(encodeString[i]) * 10 + int(encodeString[i + 1])
            # chr() 把 asc number 转换成 字符
            # 然后用 + 连接
            res = res + chr(ascNumber)
            # 到目前得到 ‘ABC’， 然后用python里非常方便的 切片 来翻转它
        return res[::-1]

        ## 写法2: 直接倒序翻译，不用翻转
        # if encodeString is None:
        #     return ""
        # asciiNumber = 0
        # reveseDecodedString = ""
        # 倒序 decode
        # for i in range(len(encodeString) - 1, 0, -2):
        #     asciiNumber = int(encodeString[i - 1]) * 10 + int(encodeString[i])
        #     reveseDecodedString += chr(asciiNumber)
        # return reveseDecodedString