class Solution:

    # 领口E 423 · Valid Parentheses
    def isValidParentheses(self, s):
        "做次:2，8.29"
        stack = []
        for c in s:
            if c in '({[':
                stack.append(c)
                continue
            if stack and (c == ')' and stack[-1] == '(' or c == '}' and stack[-1] == '{' or c == ']' and stack[-1] == '['):
                stack.pop()
            else:
                return False

        return len(stack) == 0


if __name__ == '__main__':
    sol = Solution()
    ans = sol.isValidParentheses('()')
    print(ans)