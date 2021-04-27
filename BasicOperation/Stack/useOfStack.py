from Stack import Stack


'''
刷题 423 · Valid Parentheses
题意：输入一个只包含括号的字符串，判断括号是否匹配
模拟堆栈，读到左括号压栈，读到右括号判断栈顶括号是否匹配
'''
def isValidParentheses(s):
    stack = []
    for ch in s:
        # 压栈
        if ch in '([{':
            stack.append(ch)
        else:
            # 栈需非空 （非'([{'字符的话就要pop了，先检查栈是否为空，如果是空的话就不对，就 return False）
            if not stack:
                return False
            # 判断栈顶是否匹配  （既然非空，还必须匹配正确，不匹配也return False）
            if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
                return False
            # 弹栈  （前两步通过，才能走到这一步，说明是我们想要的，就可以pop了）
            stack.pop()
    return not stack  # 最后 for 循环完，stack 是空就对了



if __name__ == '__main__':
    my_stack = Stack()
    for i in range(50):
        my_stack.push(i)

    print(my_stack.is_empty())

    while not my_stack.is_empty():
        print(my_stack.pop(), end=' ')

    s = '()'
    boolS = isValidParentheses(s)
    print(boolS)




