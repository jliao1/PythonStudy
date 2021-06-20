
# Description
# 查找斐波纳契数列中第 N 个数。
#
# 所谓的斐波纳契数列是指：
#
# 前2个数是 0 和 1 。
# 第 i 个数是第 i-1 个数和第i-2 个数的和。
# 斐波纳契数列的前10个数字是：
#
# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...

class Solution:
    def fibonacci1(self, n):
        if n <= 2:
            return n - 1

        dp = [0] * 3

        # initialize
        dp[1 % 3] = 0
        dp[2 % 3] = 1

        # Use the memorized i - 1, i - 2 Fibonacci value to calculate the ith value
        # Mod index by the array size so that the ith value will override the (i - 3)th value
        for i in range(3, n + 1):
            dp[i % 3] = dp[(i - 1) % 3] + dp[(i - 2) % 3]

        return dp[n % 3]

    #这个写法空间复杂度较高
    def fibonacci2(self, n):
        res =[]
        res.append(0);
        res.append(1);
        for i in range(2, n):
            value = res[i-1] + res[i-2]
            res.append(value)
        return res[n-1]



if __name__ == '__main__':
    sol = Solution()
    res = sol.fibonacci2(6)
    print(res)