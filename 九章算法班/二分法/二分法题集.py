'''
二分法虽然查找是O(logN)
是比字典查找慢
但二分不像字典那样，需要额外空间

一般比O(n)还优的时间复杂度只能是O(logN)的二分法

二分法可以写while和recursion
什么时候用递归？面试官要求你用。或者，递归深度不太深，是logN级别的。非递归不会写，没信心写对。
用while情况？容易stackoverflow，比如linked list不能用。iterative好写，就写interactive
面试的时候dfs一般用递归实现，但二叉树遍历的dfs一般都用iterative实现，其他算法一般也用non-recursive实现
'''

class Solution:
    # lintcode 二分法模版 easy 14 · First Position of Target
    def binarySearch(self, nums, target):
        if not nums:
            return -1

        start = 0
        end = len(nums) - 1

        while start + 1 < end:  # 这种条件下不会出现死循环，当 start和end相邻时退出
            mid = (start + end) // 2  # mid是偏小那一边的

            if nums[mid] == target:
                end = mid       # 因为要找的是 Last Position of Target
            elif target < nums[mid]:
                end = mid        # 这句也可以写成 end = mid - 1，但统一写成mid方便记
            else:  # target > nums[mid]:
                start = mid      # 这句也可以写成 start = mid + 1，但统一写成mid方便记

        if nums[start] == target:
            return start

        if nums[end] == target:
            return end

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
                start = mid  # 因为要找的是 Last Position of Target
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

    # lintcode easy 457 · Classical Binary Search
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

    # lintcode easy 457 · Classical Binary Search
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

    # lintcode倍增法 Medium 447 · Search in a Big Sorted Array 倍增法Exponential backoff
    def searchBigSortedArray(self, reader, target):
        # 倍增法找右界，时间复杂度logK
        Kth = 1
        while reader.get(Kth - 1) < target:  # 为什么是 Kth-1？因为第k个数的话下标是Kth-1
            Kth = Kth * 2;  # 为什么是乘以2，不是诚意3,4？ 因为对于计算机来说，乘以2执行得更快
            #                          还有因为 log以2或3或4为底，时间复杂度都是一个level的
            #                          还因为乘以2的话，是二分，对于后面找范围 start～end 更方便，在某个半边找就可以了。不用三等分或四等分里找了
        start = 0  # start 也可以是 K / 2，但是我习惯比较保守的写法, 写为 0 也不会影响时间复杂度
        end = Kth - 1
        # 在 start ～ end里二分法找target
        while start + 1 < end:
            mid = start + (end - start) // 2

            if reader.get(mid) < target:
                start = mid
            else:  # reader.get(mid) > target 和 reader.get(mid) == target
                end = mid

        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end

        return -1

    # lintcode Medium 460 · Find K Closest Elements
    def kClosestNumbers(self, A, target, k):
        if not A:
            return A
        # 二分法找位置
        right = self.find_upper(A, target)  # 右边的一定大于等于target
        left = right - 1  # 左边的一定大于等于target

        res = []
        # 背向双指针
        for _ in range(k):
            # 复杂的逻辑判断，最好都包装成函数处理（一个函数最多别超过20行）
            if self.is_left_closer(A, target, left, right):
                res.append(A[left])
                left -= 1
            else:
                res.append(A[right])
                right += 1
        return res
    def is_left_closer(self, A, target, left, right):
        if left < 0:
            return False
        if right > len(A) - 1:
            # 如果left没越界但right越界了，就return True说明该选左边
            return True
        # left 和 right 都在界内
        return abs(target - A[left]) <= abs(A[right] - target)
    def find_upper(self, A, target):
        start = 0
        end = len(A) - 1

        while start + 1 < end:
            mid = (start + end) // 2
            # 这是找first position ≥ target的情况
            if A[mid] < target:
                start = mid
            else:
                end = mid

        if A[start] >= target:  # 这是找first position ≥ target的情况
            return start
        if A[end] >= target:
            return end

        # 找不到的情况
        return len(A)

if __name__ == '__main__':
    sol = Solution()
    res = sol.binarySearch([1,2,4,4,4,5,5], 1)
    print(res)