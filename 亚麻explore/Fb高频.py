class Solution:

    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        for i in range(len(s)):

            if s[i] == ')' and stack and s[stack[-1]] == '(':
                stack.pop()
                continue

            if s[i] in '()':
                stack.append(i)

        pass


if __name__ == '__main__':

    sol = Solution()

    res = sol.minRemoveToMakeValid("))((")

    print(res)