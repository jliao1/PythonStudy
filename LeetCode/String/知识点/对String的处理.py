'''
【知识点】

（1）ASC码是0-127位
A-Z： 65-90
a-z:  97-122

（2）小写字母 -> 大写字母：
ord() 是返回对 某单个字符的 应的ASCII 数值
      但API里的意思是 Given a string representing one Unicode character,
      return an integer representing the Unicode code point of that
      character. For example, ord('a') returns the integer 97.
chr() Return the string representing a character whose Unicode
      code point is the integer i. For example, chr(97) returns
      the string 'a'
写法一：
    lower_char = 'm'
    upper_char = chr( ord(lower_char) - ord('a') + ord('A') )
写法二：
    lower_char = 'a'
    upper_char = chr(ord(c) - 32)
写法三：
    upper_char = lower_char.upper( )

    另外，字母直接是可以相互比较的

（）判断一个字符串里 是不是 字母/数字
    str.isalpha()  # 判断是不是字母
    Return True if all characters in the string are alphabetic and
    there is at least one character, False otherwise.

    str.isdigit()  # 判断是不是数字

（3）合并连接两个字符串 + 和 join()
    str1 = " ".join(["hello", "world"])
    str2 = "hello " + "world"
    print(str1)  # 输出 “hello world"
    print(str2)  # 输出 “hello world"

    str = "-";
    seq = ("a", "b", "c"); # 字符串序列
    print str.join( seq ); # 输出 a-b-c


    join的运算效率高于 +, 为什么？
    【+】
    由于字符串是不可变对象，当使用“+”连接字符串的时候，
    每执行一次“+”操作都会申请一块新的内存，然后复制上一
    个“+”操作的结果和本次操作的有操作符到这块内存空间中，
    所以用“+”连接字符串的时候会涉及内存申请和复制；
    【join()】
    在连接字符串的时候，首先计算需要多大的内存存放结果，
    然后一次性申请所需内存并将字符串复制过去。在用"+"连接字
    符串时，结果会生成新的对象，而用join时只是将原列表中的
    元素拼接起来，因此在连接字符串数组的时候会考虑优先使用join。

（4）str ='ab'
    str_len = len(str)

（5）取字串
    str[i : j]   是范围 [i,j) 不包含j

（6）查找某个字串
    str = 'abcdef'
    index = str.find('abc')  如果不存在 返回 -1， 如果存在返回 sub字串的第一次出现的index

（7）强制类型转换
    number -> string : str(number)
    string -> number : int(str)
                       float(str)
    但这种会报错：
    s = 'a'
    int(s)  # 因为s根本就不是数字，无法转换成数字

（8）翻转字符串
    s = "abcdefg"
    s[::-1]

（9）字符串切片
    讲解一：比如
     0  1  2  3 4 5 6 7 8 9 10 11 12
    "a  b  c  d e f g h i j  k  l m"
    -12-11-10-9-8-7-6-5-4-3 -2 -1 0
    'abcdefghijklm' [2:10:3]  # start at 2, go up to 10, count by 3 是 'cfi' （index是2,5,8）
    'abcdefghijklm' [10:2:-1] # start at 10, go down to 2(但不包括2), count down by 1 是 'kjihgfed' (index从10位倒数到3位)
    'abcdefghijklm' [::3]  # beginning to end, counting by 3 是 'adgjm' (index是0,3,6,9,12)
    'abcdefghijklm' [::-3] # end to beginning, counting down by 3 是 'mjgda' (index是12,9,6,3,0)

    讲解二：
    >>>  a='0123456'
    >>> a[1:5]'1234'
    返回结果是1234，能理解么？
    第一位到第五位切片（初始是0位）

    >>> a[1:5:2]
    '13'
    这个呢？能理解么？1-5位切片，步进为2，所以取出来是‘13’
    那么问题就来了[::-1]表示的是从头到尾，步长为-1，你感受一下。

    讲解三：
    Sequence[start:end:step]
    step的正负决定了切片的方向。
    step为正，左 -> 右，若start > end，结果为空，因为start右边无值
    step为负，右 -> 左，若start < end，结果为空，因为start左边无值
    若start为空，则表示最开始位置
    若end为空，则表示到最后位置
    若start和end都为空，则表示全部元素。

（10）两种方法分割python多空格字符串
    str = " aa   bbbbb         ccc  d"

    这种不行：
    str_list2 = str.split(' ')
    print(str_list2)  # 打印出来是 ['', 'aa', '', '', 'bbbbb', '', '', '', '', '', '', '', '', 'ccc', '', 'd']

    这种可以：
    str_list1 = str.split()
    print(str_list1)  # 打印出来是 ['aa', 'bbbbb', 'ccc', 'd']

【字符串之间比较】
    if A == B 是判断 两个字符串内容是否相等

'''