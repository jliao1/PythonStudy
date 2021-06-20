# Leetcode：Reverse ASCII Encoded Strings
# 思路：每两个数字是一对地翻译，比如 字符 ‘6’‘5’是一对 -> 65 -> 'A'   最后切片翻转字符串  或者一开始就可以倒序翻译
#      Why两个数字作为一对翻译呢？ 好像是因为在Asc码中，大写是从65到90，所以是两位一对




class Solution:
    """
    @param encodeString: an encode string
    @return: a reversed decoded string
    """

    def reverseAsciiEncodedString(self, encodeString):
    #
    #     # 写法1：每两个数字是一对地翻译，比如 字符 ‘6’‘5’是一对 -> 65 -> 'A'   最后切片翻转字符串
    #     if encodeString is None:
    #         return ""
    #
    #     res = ""  # 用来存储
    #
    #     for i in range(0, len(encodeString), 2):  # 每两个两个数字地取
    #         # 比如 字符 ‘6’ ‘5’ 是一对 -> 65 -> 'A'
    #         # string 强制转换 int（知识点）
    #         ascNumber = int(encodeString[i]) * 10 + int(encodeString[i + 1])
    #         # chr() 把 asc number 转换成 字符  （知识点）
    #         # 然后用 + 连接  （知识点）
    #         res = res + chr(ascNumber)
    #         # 到目前得到 ‘ABC’， 然后用python里非常方便的 切片 来翻转它
    #     return res[::-1]  # 切片翻转字符串（知识点）

        # 写法2: 直接倒序decode，不用翻转，
        # 九章老师上课讲的写法
        if encodeString is None:
            return ""

        asciiNumber = 0
        res = ""
        # 倒序 decode
        for i in range(len(encodeString) - 1, 0, -2):
        # 利用切片截取 i-1 到 i 的string, 注意切片(inclusive, exclusive)
            asciiNumber = int( encodeString[(i-1):(i+1)]  )
            # 把ascii Number 用 int 强制转换成数字
            res += chr(asciiNumber)
        return res

