# Leetcode：Reverse ASCII Encoded Strings
# 思路：每两个数字是一对地翻译，比如 字符 ‘6’‘5’是一对 -> 65 -> 'A'   最后切片翻转字符串  或者一开始就可以倒序翻译
#      Why两个数字作为一对翻译呢？ 好像是因为在Asc码中，大写是从65到90，所以是两位一对

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
            # string 强制转换 int（知识点）
            ascNumber = int(encodeString[i]) * 10 + int(encodeString[i + 1])
            # chr() 把 asc number 转换成 字符  （知识点）
            # 然后用 + 连接  （知识点）
            res = res + chr(ascNumber)
            # 到目前得到 ‘ABC’， 然后用python里非常方便的 切片 来翻转它
        return res[::-1]  # 切片翻转 （知识点）

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


    '''
    还有一些 知识点
    
    小写字母转大写字母：
    写法一：
        lower_char = 'm'
        upper_char = chr( ord(lower_char) - ord('a1 ) + ord('A') )
    写法二：
        upper_char = lower_char.upper( )
        
    合并连接两个字符串 +
    print( 'ab' + 'cd')
    
    求字符串长度 
    str ='ab'
    str_len = len(str)
    
    取字串
    str[i : j]
    
    查找某个字串
    str = 'abcdef'
    index = str.find('abc')  如果不存在 返回 -1， 如果存在返回 字串的首位置
    
    强制类型转换
    number -> string : str(number)
    string -> number : int(str)
                       float(str)
                       
    字符串切片
    （1）讲解一：比如
     0  1  2  3 4 5 6 7 8 9 10 11 12
    "a  b  c  d e f g h i j  k  l m"
    -12-11-10-9-8-7-6-5-4-3 -2 -1 0
    'abcdefghijklm' [2:10:3]  # start at 2, go up to 10, count by 3 是 'cfi' （index是2,5,8）
    'abcdefghijklm'[10:2:-1] # start at 10, go down to 2(但不包括2), count down by 1 是 'kjihgfed' (index从10位倒数到3位)
    'abcdefghijklm'[::3]  # beginning to end, counting by 3 是 'adgjm' (index是0,3,6,9,12)
    'abcdefghijklm'[::-3] # end to beginning, counting down by 3 是 'mjgda' (index是12,9,6,3,0)
    （2）讲解二：
    >>>  a='0123456'
    >>> a[1:5]'1234'
    返回结果是1234，能理解么？
    第一位到第五位切片（初始是0位）
    
    >>> a[1:5:2]
    '13'
    这个呢？能理解么？1-5位切片，步进为2，所以取出来是‘13’
    那么问题就来了[::-1]表示的是从头到尾，步长为-1，你感受一下。
    （3）讲解三：
    Sequence[start:end:step]
    step的正负决定了切片的方向。
    step为正，左 -> 右，若start > end，结果为空，因为start右边无值
    step为负，右 -> 左，若start < end，结果为空，因为start左边无值
    若start为空，则表示最开始位置
    若end为空，则表示到最后位置
    若start和end都为空，则表示全部元素。

    
    '''