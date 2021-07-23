# 九章算法：Jump Game, 05032021  没懂

# 时间复杂度：O(n)，n为A长度。一次遍历。
# 空间复杂度：O(1)

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