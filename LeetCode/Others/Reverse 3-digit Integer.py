# 考点  熟练用 /  //  % 这几个符号

class Solution:
    """
    @param number: A 3-digit number.
    @return: Reversed number.
    """
    def reverseInteger(self, number):
        # write your code here
        return number % 10 * 100 + number // 10 % 10 * 10 + number // 100