# Description
# Given number n. Print number from 1 to n. According to following rules:
#
# when number is divided by 3, print "fizz".
# when number is divided by 5, print "buzz".
# when number is divided by both 3 and 5, print "fizz buzz".
# when number can't be divided by either 3 or 5, print the number itself.
# Example
# Example 1:
#
# Input:
#
# n = 15
# Output:
#
# [
#   "1", "2", "fizz",
#   "4", "buzz", "fizz",
#   "7", "8", "fizz",
#   "buzz", "11", "fizz",
#   "13", "14", "fizz buzz"
# ]



class Solution:
    """
    @param n: An integer
    @return: A list of strings.
    """
    def fizzBuzz(self, n):
        # write your code here
        res = []
        for i in range (1, n+1):
            # 分析题要仔细，这是有优先级打印的，被15整除的优先打印
            if i % 3 ==0 and i % 5 == 0:
                res.append("fizz buzz")
            elif i % 3 ==0:
                res.append("fizz")
            elif i % 5 ==0:
                res.append("buzz")
            elif i % 3 !=0 and i % 5 != 0:
                res.append(str(i))  # str(i)是把数字i强制转换成string
        return res

    #三目运算符解法
    # a = 5
    # b = 3
    # st = "a大于b" if a > b else "a不大于b"
    # # 输出"a大于b"
    # print(st)
    def fizzBuzz(self, n):
        # write your code here
        res = []
        for i in range (1, n+1):
            (res.append("fizz buzz") if i % 5 == 0 else res.append("fizz")) if i % 3 == 0 else (res.append("buzz") if i % 5 == 0 else res.append(str(i)))
        return res