def calculate1( a, operator, b):
    # write your code here
    switcher = {
        '+': addition,
        '-': subtraction,
        '*': mul,
        '/': division
    }
    method = switcher.get(operator, None)
    if method:
        return method(a,b)

def calculate2( x, operator, y):
    # write your code here
    return {
        '+': lambda: x + y,
        '-': lambda: x - y,
        '*': lambda: x * y,
        '/': lambda: x / y,
    }.get(operator, lambda: None)()


def addition(num1, num2):
    num1 += num2
    return num1


def subtraction(num1, num2):
    num1 -= num2
    return num1


def mul(num1, num2):
    num1 *= num2
    return num1


def division(num1, num2):
    num1 /= num2
    return num1




if __name__ == "__main__":
    a = calculate2(1,'/',2)
    print(a)