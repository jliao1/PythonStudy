# 用python字典的形式实现 switch case 语句

# Description
# Given two integers a and b, an operator, choices:
#
#     +, -, *, /
# Calculate a <operator> b.




class Calculator:
    """
    @param x: An integer
    @param operator: A character, +, -, *, /.
    @param y: An integer
    @return: The result
    """
    def calculate(self, x, operator, y):
        # write your code here
        return {
            '+': lambda: x + y,
            '-': lambda: x - y,
            '*': lambda: x *  y,
            '/': lambda: x /  y,
            }.get(operator, lambda: None)()