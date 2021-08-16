'''
二分法虽然查找是O(logN)
是比字典查找慢
但二分不像字典那样，需要额外空间

一般比O(n)还优的时间复杂度只能是O(logN)的二分法

二分法可以写while和recursion
什么时候用递归？面试官要求你用。或者，递归深度不太深，是logN级别的。非递归不会写，没信心写对。
用while情况？容易stackoverflow，比如linked list不能用。iterative好写，就写interactive
面试的时候dfs一般用递归实现，但二叉树遍历的dfs一般都用iterative实现，其他算法一般也用non-recursive实现

【四重境界】
第二重境界：
将（一般是已经排好序，或者有规律的）数组分成两个部分
满足某个条件为一部分，不满足某个条件为另一个部分
左边OOO，右边XXX
让你找OOO数组中最后一个，或者XXX中第一个满足某个条件的位置
但注意，如果有重复的数，无法保证在logN的时间复杂度哪解决
      比如[1,1,1,……,1,1]中间藏着一个0，还是需要把每个位置上的1都for看一遍才能找到0，不然没有办法判断左边还是右边大

第三重境界：不带sorted属性
无法找到一个条件形成OOXX模型

第四重境界：
确定答案范围，验证答案大小
可能就无法通过时间复杂度O(logN)来猜了
但二分反正一定是去找输入/输出，分析出OOXX的特性，去找最后一个O和第一个X
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

            if target >= nums[mid]:
                start = mid
            else:
                end = mid
            ''' 以上四行等于
            if nums[mid] == target:
                end = mid       # 因为要找的是 Last Position of Target
            elif target < nums[mid]:
                end = mid        # 这句也可以写成 end = mid - 1，但统一写成mid方便记
            else:  # target > nums[mid]:
                start = mid      # 这句也可以写成 start = mid + 1，但统一写成mid方便记
            '''
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

    # 第二重境界简化成OOXX模型 lintcode Medium 585 · Maximum Number in Mountain Sequence
    def mountainSequence(self, nums):
        if not nums:
            return -1

        left = 0
        right = len(nums) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] >= nums[mid + 1]:
                right = mid
            if nums[mid] < nums[mid + 1]:
                left = mid

        return max(nums[left], nums[right])

    # lintcode Medium 159 · Find Minimum in Rotated Sorted Array 继续 OOXX模型化
    def findMin(self, nums):
        """
        一般不特别指明，默认的sorted array就是从 小 -> 大
        rotated array 就是循环右移,或循环左移
        sorted array 是 rotated sorted array的一个中间产物
        sorted array 是 ∈ rotated sorted array 的，二分后子问题也是跟原问题一样是 rotated sorted array
        看到一些数据，具像化在脑子里
         / ｜
        /  ｜          图形结构是这样
        -------------   哐叽一下掉下来
            ｜   /
            ｜  /       找的是右半部分第一个值 ≤ 最后一个值
        OOOO  XXX
        """
        if not nums:
            return -1
        start = 0
        end = len(nums) - 1

        target = nums[-1]

        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] <= target:
                end = mid   # 向左找
            else:
                start = mid  # 向右找

        return min(nums[start], nums[end])

    # lintcode Medium 62 · Search in Rotated Sorted Array 继续 OOXX模型化
    def search1(self, A, target):
        """
        方法1：可以像159题那样先找到最小值所在的位置，然后再在最小值左侧或者右侧去找target。用了2次二分法
        如果面试官继续让你只用一次二分法做，该怎么做？看方法2
        """
        if not A:
            return -1

        min_index = self.find_min(A)

        if A[min_index] <= target <= A[-1]:
            # 右下部分
            return self.binary_search(A, target, min_index, len(A) - 1)
        elif A[0] <= target <= A[min_index-1]:
            # 左上部分
            return self.binary_search(A, target, 0, min_index-1)
        else:
            # 右下/左上 都没找到
            return -1
    def binary_search(self, A, target, start, end):
        while start + 1 < end:
            mid = (start + end) // 2
            if target > A[mid]:
                start = mid
            else:
                end = mid

        if A[start] == target:
            return start

        if A[end] == target:
            return end

        return -1
    def find_min(self, A):
        start = 0
        end = len(A) - 1

        target = A[-1]

        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] <= target:
                end = mid
            else:
                start = mid

        if A[start] <= A[end]:
            return start
        else:
            return end

    # lintcode Medium 62 · Search in Rotated Sorted Array 继续 OOXX 模型化
    def search2(self, A, target):
        """
        二分完了后，对象的属性是不能变的。二分之前是一个rotated sorted array，sorted array也是rotated sorted array的
        回归二分法的本质，切一刀mid
        主要是看mid在左上部分，还是右下部分，是可以判断的  这就分成4种情况判断了
        """
        if not A:
            return -1

        start, end = 0, len(A) - 1
        while start + 1 < end:
            # 先切一刀mid
            mid = (start + end) // 2

            if A[mid] >= A[start]:
                # 如果 mid 在左上角
                # 左范围是
                if A[start] <= target <= A[mid]:  # 做范围是个sorted array 也是 rotated sorted array
                    end = mid   # 往左边去
                else:  # 右范围是 大于mid 或 小于 start，也是一个上升然后掉下来的 rotated sorted array
                    start = mid  # 往右边去
            else:
            # 不然 mid位置 就是在右下角
                # 右范围是
                if A[mid] <= target <= A[end]:
                    start = mid
                else: # 左范围是 大于start 或小于mid
                    end = mid

        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1

    # 三重境界 lintcode Medium 75 · Find Peak Element
    def findPeak(self, A):
        """
        如果要找所有peak，worst case下是每隔1个有一个peak，最多有n/2个peak，不可能比O(n)更快
        如果找最大的peak，在这个无序的数组里，二分后也无法判断去左边还是去右边，也就只有从前到后打擂台了
        但这道题是return any peak
        （1）相邻元素不相等
        （2）一定先上升，最后下降
        这特殊条件带来的启示是什么？先升后降低必有peak
        而这道题要从O(n)到O(logN)那就应该是二分法，切一刀，切一刀后怎么比呢？就用到上面的特点
        留下的半部分，也必须保证是先升后降才行
        如果mid在上坡路，说明mid右侧必有peak
        如果mid在下坡路，说明mid左侧必有peak
        如果mid在谷底，说明左右都有peak，去左右找都无所谓
        如果mid在peak，就return
        """
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            # 四种情况搞成三种就可以覆盖所有啦
            if A[mid] < A[mid - 1]:
                end = mid
            elif A[mid] < A[mid + 1]:
                start = mid
            else:
                # 这种情况说明 mid 比左右都大，是peak啦
                return mid

        # 这个代码不能保证返回的是第一个peak的
        if A[start] < A[end]:
            return end
        else:
            return start

    # 四重境界 lintcode hard 183 · Wood Cut
    def woodCut(self, L, k):
        """
        The longest length = 1, 2, 3, 4, 5, 6……
                             O  O  O  O  X  X ……不是基于原数组的，而是基于答案取值范围的
                           k满足条件的打O｜开始不满足了打XX
                            求的是OOO部分的last pisition of 满足 something

        这个时间复杂度不是O(logN),虽然二分是O(logN)
        但是get_pieces(L, mid)是O(N)
        所以总的是 O(NlogSum)
        """
        if not L:
            return 0

        start = 1
        #         end有两个约束，取个更短的
        end = min(max(L), sum(L) // k)

        while start + 1 < end:
            mid = (start + end) // 2
            if self.get_pieces(L, mid) >= k:
                # 满足，说明是OO
                # 接着往右边找，我们要找的是OOO的last position
                start = mid
            else:  # 不满足的是XXX，往左边去
                end = mid

        if self.get_pieces(L, end) >= k:
            # 先看end，因为我们要返回的是last position of OOO
            return end
        if self.get_pieces(L, start) >= k:
            return start

        return 0
    def get_pieces(self, L, length):
        return sum(l // length for l in L)


    # 四重境界 OA题 Global maximum 题目 https://leetcode.com/discuss/interview-question/1215681/airbnb-oa-global-maximum
    def findMaximum(self, A, K):
        """时间复杂度O[Nlog(end)] 或者 O(NlogN)， 看 end 和 N谁更大了"""
        n = len(A)
        A.sort()  # 这个时间复杂度是O(NlogN)

        # 注意这个start和end是答案范围，在此范围里二分
        start = 0
        end = A[n - 1] - A[0]

        while start + 1 < end:
            mid = (start + end) // 2
            # check whether it is possible to get a subString of size K with a minimum difference among any pair equal to mid.
            # 是否存在一个是 K size 的 sub_array, 这个 sub_array 里 两两的差，最小是 mid
            if self.exist_sub_of_size_k(A, K, mid):
                # OO：如果符合条件
                # 去right half 找，因为这题要求的 mid 尽量大
                start = mid
            else:
                # XX：如果不如何条件，K size 的sub-array，里面元素两两差，最小是 mid的，没有
                # 说明是 取值 太大导致的，就往left half去找，取值小点儿
                end = mid

        if self.exist_sub_of_size_k(A, K, end):
            return end
        if self.exist_sub_of_size_k(A, K, start):
            return start

        return res
    def exist_sub_of_size_k(self, array, K, min_difference):
        """
        看 array 里 是否存在一个 sub-array，它的 minimum difference among any pair equal to "min_difference"
        这个 sub-array 里的元素的两两之差, 最小的差是 "min_difference"
        如果是的话，返回 true; 找不到的话返回 False
        时间复杂度O(n)
        """
        # 注意, 传进来的array已经是sort好的了

        len_of_sub = 1
        sub = [array[0]]
        last = array[0]

        for i in range(1, len(array)):  # i 从第二个数开始
            '''
            因为 array 已经是 sorted 的从 小->大的
            在pick元素的时候，只要 这个array[i] 与 last 之差大于 min_difference
            就能保证 shortest_sub 里俩俩之差，最小的是 min_difference 啦
            '''
            if array[i] - last >= min_difference:
                # 说明 array[i] 符合要求，pick 它!
                sub.append(array[i])
                # 把last更新一下
                last = array[i]
                len_of_sub += 1
                # 每pick完一个元素，检查一下是否符合想要的K长度，是的话就立刻返回, 这样才能保证找到符合要求的sub是shortest的
                if (len_of_sub == K):
                    print("shortest_sub:", sub)  # 这条打印是为了测试用的
                    return True
        return False

if __name__ == '__main__':
    sol = Solution()
    A = [1,2,3,4]
    m = 3
    res = sol.findMaximum(A, m )
    print(res)