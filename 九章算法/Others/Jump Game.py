# Description
# Given an array of non-negative integers, you are initially positioned at the first index of the array.
#
# Each element in the array represents your maximum jump length at that position.
#
# Determine if you are able to reach the last index.
#
# The array A contains 𝑛 integers 𝑎1, 𝑎2, …, 𝑎𝑛 (1≤𝑎𝑖≤5000) (1≤n≤5000 )
#
# Example
# Example 1:
#
# Input:
#
# A = [2,3,1,1,4]
# Output:
#
# true


# 贪心法
class Solution:
    """
    @param A: A list of integers
    @return: A boolean
    """
    def canJump(self, A):
        n, rightmost = len(A), 0
        # 依次遍历每个位置i
        for i in range(n):
            # 如果i在rightmost范围内，说明i可达
            if i <= rightmost:
                # 将rightmost更新为max(rightmost, i + A[i])
                rightmost = max(rightmost, i + A[i])
                # 如果rightmost大于等于数组中的最后一个位置，那就说明最后一个位置可达
                if rightmost >= n - 1:
                    # 就可以直接返回True
                    return True
        # 若在遍历结束后，最后一个位置仍然不可达，就返回 False
        return False