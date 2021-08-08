'''
二分法虽然查找是O(logN)
是比字典查找慢
但二分不像字典那样，需要额外空间
'''

class Solution:

    # lintcode easy 457 · 经典二分查找问题
    def findPosition_recursive(self, nums, target):
        def binary_search(num, target, start, end):
            if start > end:
                return -1

            mid = (start + end) // 2

            if target == nums[mid]:
                return mid

            if target > nums[mid]:
                return binary_search(num, target, mid + 1, end)

            # 代码能走到这里就是 if target < nums[mid] 情况
            return binary_search(num, target, start, mid - 1)

        if not nums:
            return -1

        return binary_search(nums, target, 0, len(nums) - 1)

    # lintcode easy 457 · 经典二分查找问题
    def findPosition_iterative(self, nums, target):
        """
        但这个解答，如果一个数组内有相同数字，返回的不确定是第几个
        """
        if not nums:
            return -1

        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = (start + end) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                start = mid + 1
            else:
                end = mid - 1

        return -1

    # lintcode easy 458 · Last Position of Target
    def lastPosition_wrong_answer(self, nums, target):
        """
        这种写法会超时，因为无法处理 [1, 1] 这种情况 start = mid = 0 会无限循环下去
        办法就是，那就不要让 while 直到 start 和 end 相当再退出
        """
        if not nums:
            return -1

        start = 0
        end = len(nums) - 1
        while start < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid + 1
            else: # nums[mid] > target:
                end = mid - 1

        if nums[start] == target:
            return start

        return -1

    # lintcode easy 458 · Last Position of Target
    def lastPosition(self, nums, target):

        if not nums:
            return -1

        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid + 1  # 这句也可以写成 start = mid 对正确性来说没问题
            else:  # nums[mid] > target:
                end = mid - 1    # 这句也可以写成 end = mid

        # 先比较 end
        if nums[end] == target:
            return end

        if nums[start] == target:
            return start

        return -1

    # lintcode easy 14 · First Position of Target 二分法模版
    def binarySearch(self, nums, target):
        if not nums:
            return -1

        start = 0
        end = len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2

            if nums[mid] == target:
                end = mid
            elif nums[mid] > target:
                end = mid        # 这句也可以写成 end = mid - 1
            else:  # nums[mid] < target:
                start = mid      # 这句也可以写成 start = mid + 1

        if nums[start] == target:
            return start

        if nums[end] == target:
            return end

        return -1

if __name__ == '__main__':
    sol = Solution()
    res = sol.binarySearch([1,2,4,4,4,5,5], 1)
    print(res)