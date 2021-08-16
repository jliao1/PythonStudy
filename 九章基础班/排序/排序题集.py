class Solution:

    # lintcode Easy 479 · Second Max of Array
    def secondMax(self, nums):
        """用打擂台的方式找出第二大值"""
        maxVal = max(nums[0], nums[1])
        secVal = min(nums[0], nums[1])

        # 从第3个数组也就是index=2 开始扫
        for i in range(2, len(nums)):
            if nums[i] > maxVal:
                # 如果 nums[i] 比 maxValue 都大
                # 那就把 maxVal 和 secVal 都依次 update 一下
                secVal = maxVal
                maxVal = nums[i]
            elif nums[i] > secVal:
                # 如果 nums[i] 只比 secVal 大，那就只更新 secVal
                secVal = nums[i]

        return secVal

    # lintcode Easy 1200 · Relative Ranks
    def findRelativeRanks(self, nums):
        # sorted_Score 是用来降序排列的
        sorted_Score = sorted(nums, reverse=True)
        dic = {}

        # 搞了个字典，每个数字 该 对应第几名
        for i, num in enumerate(sorted_Score):
            if i == 0:
                dic[num] = "Gold Medal"
            elif i == 1:
                dic[num] = "Silver Medal"
            elif i == 2:
                dic[num] = "Bronze Medal"
            else:
                dic[num] = str(i + 1)

        return [dic[each] for each in nums]

    # lintcode Easy 1200 · Relative Ranks
    def findRelativeRanks2(self, nums):
        '''这种是tuple写法，有点绕，感觉没必要但也挺有意思'''
        def rank_to_medal(rank):
            if rank == 1:
                return 'Gold Medal'
            elif rank == 2:
                return 'Silver Medal'
            elif rank == 3:
                return 'Bronze Medal'
            else:
                return str(rank)

        # 这种写法挺有意思！！包装成了元组数组
        with_index = [(nums[i], i) for i in range(len(nums))]

        # 这种写法挺有意思 sort by score
        with_index.sort(key=lambda x: x[0], reverse=True)
        # 或上面这句写成 with_index.sort(reverse=True)

        dict = {}
        for rank in range(1, len(with_index) + 1):
            dict[with_index[rank - 1][1]] = rank_to_medal(rank)

        # 这种写法挺有意思
        temp = [dict[i] for i in range(len(nums))]
        return temp

    # lintcode Easy 173 · Insertion Sort List
    def insertionSortList(self, head):
        """
        思路：
        1.先特判空链表情况
        2.判断待排的节点是否需要插入
        3.从head开始找到合适的插入位置进行插入，并更新待排节点
        时间复杂度 O(n) 空间复杂度 O（1）
        """
        if not head:
            return head

        dummy = ListNode(-float('inf'), head)
        cur = head
        while cur and cur.next:
            # 先判断cur.next是不是需要 往前插的
            if cur.next.val < cur.val:
                # 如果是先把这 node 从链表中取出后删除
                insert_node = cur.next
                cur.next = cur.next.next

                # 再从头扫, 看拿出来的 node 需要插在哪里
                start = dummy  # 建立dummy的好处就在这里，把头部情况统一成general
                while start and start.next:
                    # 找到插入位置后，插进入
                    if insert_node.val < start.next.val:
                        insert_node.next = start.next
                        start.next = insert_node
                        break
                    start = start.next
            else:
                # cur.next 不需要往前插就继续往后
                cur = cur.next

        return dummy.next

    # lintcode Easy 6 · Merge Two Sorted Arrays
    def mergeSortedArray(self, A, B):
        if not A:
            return B
        if not B:
            return A

        la, lb = len(A), len(B)
        a, b = 0, 0
        res = []

        for k in range(la+lb):
            # 这个条件要注意一下哈
            # A有可取的     B没可取的 或  A元素小于等于B
            if a < la and (b >= lb or A[a] <= B[b]):
                res.append(A[a])
                a += 1
            else:  # 其他情况取B中元素
                res.append(B[b])
                b += 1

        return res

    # lintcode Medium 49 · Sort Letters by Case 其实就是quick sort一次，我觉得这题算easy。比较牛逼的是解法2
    def sortLetters1(self, chars):
        left = 0
        right = len(chars) - 1

        while left <= right:
            while left <= right and chars[left] >= 'a' and chars[left] <= 'z':
                left += 1

            while left <= right and chars[right] >= 'A' and chars[right] <= 'Z':
                right -= 1

            if left <= right:
                chars[left], chars[right] = chars[right], chars[left]
                right -= 1
                left += 1

        return chars

    # lintcode Medium 49 · Sort Letters by Case 一句话的解法
    def sortLetters2(self, chars):
        # 出来的结果是小写在前，大写在后
        chars.sort(key=lambda c: c.isupper())
        pass

    # lintcode Medium 532 · Reverse Pairs
    def reversePairs(self, A):
        self.count = 0
        self.temp = [0] * len(A)
        self.merge_sort(A, 0, len(A) - 1)

        return self.count
    def merge_sort(self, A, left, right):
        if left >= right:
            return

        mid = (left + right) // 2
        self.merge_sort(A, left, mid)
        self.merge_sort(A, mid + 1, right)
        self.merge(A, left, right)
    def merge(self, A, left, right):
        size = right - left + 1
        mid = (left + right) // 2
        i_left = left  # left - mid
        i_right = mid + 1  # mid+1 - right

        # 这个直接把左/右部分都merge了，看 i_right（当然也可以看 i_left）
        for k in range(size):
            #    右边要有              左边没有了     右边要小于左边
            if i_right <= right and (i_left > mid or A[i_right] < A[i_left]):
                # 若 左部分还有
                if i_left <= mid:
                    # 这一步是加速的过程，暴力解O(N^2), 这里能加速到 O(NlogN)
                    self.count = self.count + (mid - i_left + 1)

                self.temp[k] = A[i_right]
                i_right += 1

            else:
                self.temp[k] = A[i_left]
                i_left += 1

        for k in range(size):
            A[left + k] = self.temp[k]

    # lintcode Medium 31 · Partition Array
    def partitionArray(self, nums, k):
        """
        最多扫描一遍数组，时间复杂度为O(n), 空间是O(1)
        和quicksort里的partition过程性质相似，都是不stable的
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            while left <= right and nums[left] < k:
                left += 1
            while left <= right and nums[right] >= k:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        return left

    # lintcode Medium 148 · Sort Colors 方法是 counting sort
    def sortColors1(self, nums):
        """
        这种是 counting sort
        这种时间复杂度是O(range), range最大是O(n)

        空间复杂度O(k), k = 2
        worst case情况下用来存颜色的 count_color 是O(n)，因为有可能一个颜色只出现1次
        如果想降低空间复杂度：
        array + 固定两三种元素 + O(N)时间 + O(1) 的空间 =》其实做两三次quick sort就好，看方法2
        """
        # 其实可以用 dict 来 count，但能用简单的数据结构就用list啦
        count_color = [0] * 3      # 这个需要额外的空间了
        for each in nums:
            count_color[each] += 1

        # sorted
        index = 0
        for i in range(len(count_color)):
            while count_color[i] > 0:
                nums[index] = i
                index += 1
                count_color[i] -= 1

        return nums

    # lintcode Medium 148 · Sort Colors
    def sortColors2(self, nums):
        """
        每次partition是O(n), 这里partition两次，遍历2次，也是O(n)
        时间复杂度就是O(1)啦，比方法1 好点

        如果面试官还要难为你说只允许遍历1次呢… 那请看方法3
        """
        self.partition_array(nums, 1)  # 出来的结果是，左半边部分 < 1，右半边部分 ≥ 1
        self.partition_array(nums, 2)  # 这次 partiton 把右半边1和2分开。其实这次partition可以不管前面的几个0了，不过不管也ok，反正是O(n)
    def partition_array(self, A, key):
        """出来的结果是，左部分 <k, 右部分 ≥ k """
        smaller_than_k = -1
        for i in range(len(A)):
            # 小于k的时候才处理，否则 它本来就≥k，不用管
            if A[i] < key:
                smaller_than_k += 1
                A[smaller_than_k], A[i] = A[i], A[smaller_than_k]

    # lintcode Medium 148 · Sort Colors
    def sortColors3(self, nums):
        """
        如果只允许遍历1次数组？
        那就0往左边丢，2往右边丢，1就在中间了
        这是个三指针做法

        但这个方法虽然只遍历了1次数组，但每步里的操作次数变多了，总的来讲操作次数跟方法2是一样的
        """
        left = -1
        right = len(nums)
        index = 0

        # i < p2 是个很重要的条件
        while index < len(nums) and index < right:
            if nums[index] == 0:
                left += 1
                nums[left], nums[index] = nums[index], nums[left]
                '''
                为啥这里没有 index -= 1
                因为交换后的 nums[index] 只可能是 1
                不需要 回溯 交换了
                所有的0也都小于p0增1前的位置
                '''
            elif nums[index] == 2:
                right -= 1
                nums[right], nums[index] = nums[index], nums[right]
                index -= 1

            index += 1

    # lintcode Medium 148 · Sort Colors
    def sortColors4(self, nums):
        """
        思路
        如果面试官让你值partition一次，思路是，可以看方法4，也类似于方法3的三指针
           _ _ _ _ _ _ _ _
          L               R
        L左侧不包括left都是0
        R右不包括right都是2
        for循环i去扫每一个数
            如果看到0，就交换扔给左边L，去扩大L的范围(L++)
            如果看到2，就交换扔给右边R，然后扩大R的范围(R--)
                     但换回来的数字有可能是0，1，2，如果还要继续去处理如果0扔到左边，2扔到右边，1就不管
            如果看到1就不管
        最后 L～R的一定是1
        """
        pass

    # lintcode Easy 464 · Sort Integers II
    def sortIntegers2(self, A):
        def quick_sort(A, start, end):
            if start >= end:
                return

            mid = (start + end) // 2
            pivot = A[mid]
            left = start
            right = end

            while start <= end:
                while start <= end and A[left] < pivot:
                    left += 1
                while start <= end and pivot < A[right]:
                    right -= 1
                while start <= end:
                    A[left], A[right] = A[right], A[left]
                    left += 1
                    right -= 1

            quick_sort(A, start, right)
            quick_sort(A, start, left)

    # lintcode Medium 143 · Sort Colors II
    def sortColorsTwo1(self, colors, k):
        """counting sort 但这版本时间复杂度O(NlogN), 空间复杂度O(k)"""
        # counting
        count_color = {}
        for each in colors:
            if each not in count_color:
                count_color[each] = 1
            else:
                count_color[each] += 1

        # sort
        # 由于用了字典结构，color 此时元素排列是 无序的
        # 因此需要最后 sort一下，但这就需要 NlogN时间了。
        # 可以看看方法2怎么解决这个问题
        index = 0
        for k, v in count_color.items():
            while v > 0:
                colors[index] = k
                index += 1
                v -= 1

        colors.sort()

    # lintcode Medium 143 · Sort Colors II
    def sortColorsTwo2(self, colors, k):
        """
        counting sort 但这版本时间复杂度O(N), 空间复杂度O(k)
        能这么做是因为题目的限定条件是 with the colors in the order 1, 2, ... k
        所以 k 的范围只是 1～k

        但这道题让挑战时间复杂度实现O(logK) 就要想到 quick sort 的空间复杂度就是 logK 节奏了
        """
        # counting
        count_colors = [0] * (k + 1)
        for each in colors:
            count_colors[each] += 1
        # sort
        index = 0
        for i, val in enumerate(count_colors):
            while val > 0:
                colors[index] = i
                val -= 1
                index += 1
        ''' 或上6行写成: 
        index = 0
        for color, count in enumerate(counter):
            for _ in range(count):
                colors[index] = color 
                index += 1 
        '''

    # lintcode Medium 143 · Sort Colors II 彩虹排序 经典
    def sortColorsTwo3(self, colors, k):
        """
        瞎猜的话这题肯定不是O(n*k)和O(n^k)，因为这两个时间复杂度都比快排NlogN大
        一般肯定要比NlogN快，不然做题就没啥意义啦
        那么是NlogK还是KlogN呢？可以举特殊例子，
        如果k=1就不需要排序，就是O(1)
        如果k=2时就两种颜色分开，就partition一次就好是O(n)
        所以乍一猜，是O(NlogK)，然后推它算法，一般有两种说法
        （1）n * logK
            降纬 n 次 logK的操作，涉及到log级的是 heap，红黑树，wk1_二分法…… 好像不太像
        （2）logK次 的 n 次操作
             归并排序：按树的结构拆解，计算每一层的时间复杂度是O(N)，一共O(logN)层
             由此知道我们希望这题有logK层，希望 K/2  K/4 …… 1 ，每层是O(N)
                                     那怎么去把K一分为二，直到分到1呢
                                     把颜色范围一分为二（顺带把数组一分为二 ）

        【彩虹排序】解决sort颜色的问题
        这个就是所谓的彩虹排序rainbow Sort 。
        Rainbow Sort其实更像是quicksort的变种，
        我们找到需要排列的颜色中间的那一个作为基准值，
        然后进行类似quicksort的partition

        时间复杂度 O(NlogK)n是数的个数， k 是颜色数目。这是基于比较的算法的最优时间复杂度
        call stack 空间复杂度 O(logK)
        因为  每次是对KK分成左右进行递归，因此有logK 层，每层递归遍历到整个序列，长度为N

        不基于比较的话，可以用计数排序（Counting Sort）
        """
        if not colors:
            return
         # rainbow sort, k 取中间颜色
        self.raibow_sort(colors, 0, len(colors) - 1, 1, k)
    def raibow_sort(self, colors, start, end, color_from, color_to):
        #  只有一个数，或颜色只有一种
        if start >= end or color_from == color_to:
            return

        # 每次选定一个中间的颜色，这个中间的颜色用给出的k来决定，
        # 将小于等于中间的颜色的就放到左边，
        # 大于中间颜色的就放到右边，然后分别再递归左右两半
        color_mid = (color_from + color_to) // 2

        left = start
        right = end

        while left <= right:
            # 移动left指针
            while left <= right and colors[left] <= color_mid:
                left += 1                   # 注意这个等号不能写在下面那个条件，因为求color_mid的整除操作是偏小的
                                            # 若等号写在下面这行，会使得partition不均匀
            while left <= right and colors[right] > color_mid:
                right -= 1

            if left <= right:  # and colors[right] <= colors[left]:
                colors[left], colors[right] = colors[right], colors[left]
                #  下面两行是可以省略的
                left += 1
                right -= 1

        self.raibow_sort(colors, start, right, color_from, color_mid)  # 好像 color_mid-1 写成 color_mid 也行
        self.raibow_sort(colors, left, end, color_mid + 1, color_to)

    # lintcode Medium 143 · Sort Colors II  时间O(2)  空间O(1)但没看懂
    def sortColorsTwo4(self, colors, k):
        """
        题目要求不使用额外的数组，一种方法是使用彩虹排序(rainbow sort)，
        但是这样虽然做到了没有使用额外的空间，但是代价是时间复杂度变成了O(N logK)，
        那么是否有方法做到时间和空间的双赢呢？
        我们重新考虑计数排序(counting sort)，这里我们需要注意到颜色肯定是 1~k ，
        那么k一定小于 colors的长度N，我们是否可以用colors自己本身这个数组呢？

        """
        size = len(colors)
        if (size <= 0): return

        index = 0
        while index < size:
            temp = colors[index] - 1

            # 遇到计数位，跳过
            if colors[index] <= 0:
                index += 1
            else:
                # 已经作为计数位
                if colors[temp] <= 0:
                    colors[temp] -= 1
                    colors[index] = 0
                    index += 1

                # 还未被作为计数位使用
                else:
                    colors[index], colors[temp] = colors[temp], colors[index]
                    colors[temp] = -1

        # 倒着输出
        i = size - 1
        while k > 0:
            for j in range(-colors[k - 1]):
                colors[i] = k
                i -= 1
            k -= 1

    # lintcode(力扣56) Medium 156 · Merge Intervals
    def merge1(self, intervals):
        """
        时间O(n)  空间O(n)
        相同思路，更精简写法看 方法3
        空间复杂度优化到O(1)请看 方法2
        """
        if not intervals:
            return []

        '''
        先让元素按照元素里的 start 升序排列一下
        可以看出，当我遍历数组的时候，每次访问一个区间ai，
        都能保证：如果ai和它前面的区间不相交，
                那么ai后面的任意区间都不能和ai前面的任意区间相交。
                这样就保证了时间复杂度为O(n)级别，即不会出现其他的遍历。
        '''
        intervals.sort( key=lambda pair: pair.start )
        # 上面的写法好像是直接在 intervals 上 sort，sort 完后 intervals 是不变的
        # 若写成这个也行 intervals = sorted(intervals, key=lambda pair: pair.start)
        #     但sort后的intervals 的地址就跟之前不一样了，是个新的对象了

        last, output = None, []
        for element in intervals:
            if not last or last.end < element.start:
                # 虽然这里把 element append进了output，
                # 但由于 last = element，后面 在修改 last 时
                # 相当于修改的是 output 里的 element，因此也会跟着变
                output.append(element)
                last = element
            else:
                last.end = max(last.end, element.end)

        return output

    # lintcode(力扣56) Medium 156 · Merge Intervals
    def merge2(self, intervals):
        """ 时间O(n) 空间O(1) """
        if not intervals:
            return intervals

        intervals.sort(key=lambda x: x.start)
        for i in range(len(intervals)):
            #                                   如果有区间重合
            if i < len(intervals) - 1 and intervals[i].end >= intervals[i + 1].start:
                # 把 intervals[i + 1] 的区间变得更广阔了
                intervals[i + 1].start = intervals[i].start
                intervals[i + 1].end = max(intervals[i].end, intervals[i + 1].end)
                # 而 intervals[i] 不要了
                intervals[i] = None

        # 踢掉 None 的元素
        intervals[:] = [item for item in intervals if item]

        return intervals

    # 力扣56 Medium 156 · Merge Intervals
    def merge3(self, intervals):

        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    # lintcode(力扣179) Medium 184 · Largest Number
    def largestNumber1(self, array):
        """3种方法都思路都一样，就是写法不同"""
        string = []
        # 把整型转换成字符串
        for i in array:
            string.append(str(i))
        # 按最优策略排序， compare的方法是自己写的（比较相连后的字符串谁更大）
        import functools
        string.sort(key=functools.cmp_to_key(self.compare))

        res = ''.join(string)
        return '0' if res[0] == '0' else res
    def compare(self, a, b):
        if a + b > b + a:
            return -1
        return 1

    # lintcode(力扣179) Medium 184 · Largest Number
    def largestNumber2(self, nums):
        from functools import cmp_to_key
        nums.sort(key=cmp_to_key(lambda x, y: 1 if str(x)+str(y) < str(y)+str(x) else -1))
        if nums[0] == 0:
            return '0'
        return "".join([str(x) for x in nums])

    # lintcode(力扣179) Medium 184 · Largest Number
    def largestNumber3(self, nums):
        """
        这个的思路是，好像要俩俩前后相连比较一下
        比如 3 和 30，要 330 和 303 比较一下

        没有看懂key那里

        来分析下复杂度
        【时间】 Although we are doing extra work in our comparator,
        it is only by a constant factor.
        Therefore, the overall runtime is dominated by the
        complexity of sort, which is O(NlogN) in Python and Java.
        【空间】 we allocate O(n) additional space to store 排序后的列表和连结后的string
        """
        map(str, nums).sort(key = LargerNumKey)

        # Once the array is sorted, the most "signficant" number will be at the front
        l = sorted(map(str, nums), key = LargerNumKey)
        res = ''.join(l)

        # There is a minor edge case that comes up
        # when the array consists of only zeroes,
        # so if the most significant number is 00,
        # we can simply return 00.
        # Otherwise, we build a string out of the sorted array and return it.
        return '0' if res[0] == '0' else res

    # lintcode Medium 139 · Subarray Sum Closest
    def subarraySumClosest1(self, nums):
        """我这个解法超时了，O(n^2)"""
        dic = {-1: 0}
        prefix_sum = 0
        for i, num in enumerate(nums):
            prefix_sum += num
            dic[i] = prefix_sum

        min_sum = float('inf')
        min_range = [0, 0]
        for i in range(len(nums)):
            # j从 i-1 到 0为止
            j = i - 1
            while j >= -1:
                temp_sum = abs(dic[i] - dic[j])

                if temp_sum == min_sum and (j + 1) <= min_range[0]:
                    min_range = [j + 1, i]

                if temp_sum < min_sum:
                    min_sum = temp_sum
                    min_range = [j + 1, i]

                j -= 1

        return min_range

    # lintcode Medium 139 · Subarray Sum Closest
    def subarraySumClosest2(self, nums):
        """
        这是令狐冲版本思路是：
        先对数组求一遍前缀和，然后把前缀和排序
        题目要求子数组的和最接近0，也就是排序后的数组两个值相减最接近0
        排完序后，我们只要找相邻元素做差就好了

        前缀和预处理O(N)
        排序O(NlogN)
        相邻元素做差O(N)
        最终复杂度O(NlogN)
        """
        # 用list里的元素打包成元组来搜索
        # 这样一会儿就可以对 元组 进行sort
        # 把 list 里的元组 排列成 有序的（如果是字典的话，字典是无序的，没法儿排）
        prefix_sum = [(0, -1)]
        for i, num in enumerate(nums):
            prefix_sum.append((prefix_sum[-1][0] + num, i))

        # 这种sort好像就是默认以元组的第一个元素sort，小的在前
        prefix_sum.sort()
        closest, answer = float('inf'), []

        for i in range(1, len(prefix_sum)):
            difference = abs( prefix_sum[i][0] - prefix_sum[i - 1][0] )
            if closest > difference:
                closest = difference
                left = min(prefix_sum[i - 1][1], prefix_sum[i][1]) + 1
                right = max(prefix_sum[i - 1][1], prefix_sum[i][1])
                answer = [left, right]

        return answer

    # lintcode(力扣148) Medium 98 · Sort List 二分法处理 linkedlist
    def sortList1(self, head):
        if head is None: return None

        def getSize(head):
            counter = 0
            while (head is not None):
                counter += 1
                head = head.next
            return counter

        def split(head, step):
            i = 1
            while (i < step and head):
                head = head.next
                i += 1

            if head is None: return None
            # disconnect
            temp, head.next = head.next, None
            return temp

        def merge(l, r, head):
            cur = head
            while (l and r):
                if l.val < r.val:
                    cur.next, l = l, l.next
                else:
                    cur.next, r = r, r.next
                cur = cur.next

            cur.next = l if l is not None else r
            while cur.next is not None: cur = cur.next
            return cur

        size = getSize(head)
        bs = 1
        dummy = ListNode(0)
        dummy.next = head
        l, r, tail = None, None, None
        while (bs < size):
            cur = dummy.next
            tail = dummy
            while cur:
                l = cur
                r = split(l, bs)
                cur = split(r, bs)
                tail = merge(l, r, tail)
            bs <<= 1
        return dummy.next

    # lintcode Medium 98 · Sort List recursive写法比较好理解
    def sortList_recursive(self, head):
        """
        Top Down Merge Sort，recursive版本
        Time Complexity: O(NlogN), where N is the number of nodes in linked list.
                        The algorithm can be split into 2 phases, Split and Merge.

        Space Complexity: O(logN) , where N is the number of nodes in linked list.
                          Since the problem is recursive, we need additional space to
                          store the recursive call stack. The maximum depth of the
                          recursion tree is logN
        """
        if not head or not head.next:
            return head

        # 快慢指针找 middle
        # 下面这句就不能写成 fast, slow = head, head
        # 因为如果这样写，当还剩俩元素时，mid 就是 None，就无法继续处理了
        # 而且也无法作断开处理了
        fast, slow = head.next, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        mid = slow.next
        # 找到 middle 后，断开成 2 截
        slow.next = None

        left = self.sortList_recursive(head)
        right = self.sortList_recursive(mid)

        return self.merge(left, right)
    def merge(self, l, r):
        if not l or not r:
            return l or r
        dummy = ListNode(-1)
        temp = dummy
        while l and r:
            if l.val < r.val:
                temp.next = l
                l = l.next
            else:
                temp.next = r
                r = r.next
            temp = temp.next

        temp.next = l or r  # 竟然还有这种写法！！
                         # 这个的意思是，如果 l 非空就等于 l
                         # 如果 l 为空就等于 r

        return dummy.next

    # lintcode Medium 98 · Sort List 气屎老娘了, 难写死了
    def sortList_iterative(self, head):
        """
        The Top Down Approach for merge sort uses O(NlogN) extra space due to recursive call
        stack. 用 iterative 的方法来执行 bottom Up Merge Sort
        O(N logN)    空间 O(1)

        The Bottom Up approach for merge sort starts by splitting the problem into
        the smallest sub-problem and iteratively merge the result to solve the original problem.

        具体的算法是：
        Start with splitting the list into sub-lists of size 1. Each adjacent pair of sub-lists
        of size 1 is merged in sorted order. After the first iteration, we get the sorted lists
        of size 2. 然后 double size until we finish.
        As we iteratively split the list and merge, we have to keep track of the previous merged
        list using pointer sorted_tail and the remaining to be sorted using unsorted_remaining.
        """
        if not head or not head.next:
            return head

        length = self.get_len(head)
        sorted_dummy_head = ListNode(-1, head)
        # initialize
        split_size = 1

        while split_size < length:
            '''
            split_size先 = 1，先俩俩排序
            split_size先 = 2，再四四排序
            split_size先 = 4，再八八排序 直到最后
            这就是 Bottom Up Merge Sort
            '''
            # initialize
            unsorted = sorted_dummy_head.next
            sorted_tail = sorted_dummy_head

            while unsorted:
                first, unsorted = self.split(unsorted, split_size)
                # 返回的 unsorted 是 None 也没事，因为 split() 和 merge() 可以 handle 是None的情况
                second, unsorted = self.split(unsorted, split_size)
                sorted_tail = self.merge(sorted_tail, first, second)

            # double size
            split_size = split_size * 2

        return sorted_dummy_head.next
    def split(self, head, split_size):
        first_head = head

        i = 1
        while i < split_size and head:
            head = head.next
            i += 1

        if head is None:
            return first_head, None

        # disconnect
        second_head = head.next
        head.next = None

        return first_head, second_head
    def merge(self, sorted_tail, left, right):
        cur = sorted_tail
        while left and right:
            if left.val < right.val:
                cur.next, left = left, left.next
            else:
                cur.next, right = right, right.next
            cur = cur.next

        cur.next = left or right  # 也可以写成cur.next = l if l is not None else r
        while cur.next is not None:
            cur = cur.next
        return cur
    def get_len(self, head):
        count = 0
        p = head

        while p:
            p = p.next
            count += 1

        return count

    # lintcode Medium 5 · Kth Largest Element  可以写个例子跟着走一遍
    def kthLargestElement(self, k, A):
        """
        最容易想到的是直接排序，返回第k大的值。时间复杂度是O(nlogn)

        而这是 O(n) 的做法
        这题其实是快速排序算法的变体，
        通过快速排序算法的partition步骤，
        可以将小于等于pivot的值划分到pivot左边，
        大于等于pivot的值划分到pivot右边
        从而缩小范围继续找第k大的值(k要么在pivot左边或右边), 每次只在一半范围内查找，另一半不用查找

        平均 时间复杂度O(n) 因为 T(n) = T(n / 2) + O(n) 算出来是O(n)
        空间复杂度 O(1)
        """
        if not A or k < 1 or k > len(A):
            return None
        # 为了方便编写代码，这里将第 k 大转换成第 [len(A) - k] 小问题。
        return self.partition(A, 0, len(A) - 1, len(A) - k)
    def partition(self, nums, start, end, k):
        if start == end:
            # 说明找到了
            return nums[k]   # 也可以 return nums[start]

        left, right = start, end
        pivot = nums[(start + end) // 2]
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left, right = left + 1, right - 1

        # 情况1
        if k <= right:
            # pivot 左区间都小于等于 pivot
            return self.partition(nums, start, right, k)
        # 情况2
        if k >= left:
            # pivot 右区间都大于等于 pivot
            return self.partition(nums, left, end, k)

        # 情况3: right 和 left 中间隔了一个数，这个数就刚好是我们要找的数
        return nums[k]  # 也可以 return nums[right+1] 或 nums[left-1]

    # lintcode easy 80 · Median 领扣第5题的变形
    def median(self, nums):
        if not nums:
            return None
        return self.quick_select(0, len(nums) - 1, nums, (len(nums) - 1) // 2)
    def quick_select(self, start, end, nums, k):
        if start == end:
            return nums[start]

        left, right = start, end
        pivot = nums[(start + end) // 2]

        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left, right = left + 1, right - 1

        if k <= right:  # k和right和left都是index，小于等于right都是小于等于pivot的值
            return self.quick_select(start, right, nums, k)
        if k >= left:
            return self.quick_select(left, end, nums, k)

        return nums[right + 1]

    # lintcode Medium 606 · Kth Largest Element II
    def kthLargestElementTwo1(self, nums, k):
        # heap 简单解法 主要利用heap.nlargest取最大的K个数，最后的结果是在heap.nlargest最后的一个数。
        import heapq
        return heapq.nlargest(k,nums).pop()

    # lintcode Medium 606 · Kth Largest Element II
    def kthLargestElement2(self, nums, k):
        """
        这题没看，先记录一下
        三种解法：
        1. QuickSelect：像领口65题那样O(n)
        2. PriorityQueue：O(nlogk)
        3. Heapify：O(n + klogn) 这里写第3种
        """
        if not nums or k < 1:
            return None

        # O(n)
        nums = [-num for num in nums]
        # O(n)
        import heapq
        heapq.heapify(nums)
        ans = None
        # O(klogn)
        for _ in range(k):  # O(k)
            ans = -heapq.heappop(nums)  # O(logn)
        return ans

    # lintcode(力扣853) Medium 1477 · Car Fleet
    def carFleet(self, destination, position, speed):
        # 把车按照位置从 前 -> 后 (就是position从大到小排序)
        # zip(*iterables) 是 return an iterator of tuples
        tuple_position_speed = sorted(zip(position, speed), reverse=True)
        # 计算出每个车在无阻拦的情况下到达终点的时间
        time = [float(destination - p) / s for (p, s) in tuple_position_speed]
        # 如果后面的车到达 destination 所用的时间比前面车小，那么说明后车应该比前面的车先到
        #
        # 但是由于后车不能超过前车，所以这种情况下就会合并成一个车队，也就是说后车“消失了”。
        count = former_time = 0
        for cur_time in time:
            if former_time < cur_time:
                '''
                如果 former_time >= cur_time
                说明 后车到达 destination 所用的时间比前面车小
                说明后车该比前车先到，但是由于后车不能超前车，
                所以这种情况下，会合并成一个车队，count 不增加
                
                而当 former_time < cur_time
                上个车队到达了，这个车队隔了更久才到，它们不会相遇
                说明新的车队出现了，count 该+1
                '''
                count += 1
                former_time = cur_time
        return count


class LargerNumKey(str):
    """
    custom comparator preserves transitivit
    + 是字符串连接符

    如果 a + b > b +c
    和 b + c > c + b
    那么 a + c > c + a
    """
    def __lt__(x, y):
        return x+y > y+x

class Interval(object):
    def __init__(self, start, end):
        self.end = end
        self.start = start

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

def largestNumber_wrong(nums):
    max_len = 0
    for each in nums:
        # ！！！！注意这个写法是错误的
        # 因为 each 是 int 是 str
        # 这样赋值是不能改变原 nums 里的 each的
        # 左边的each是个str了，右边的each是int
        # 由于是不可变对象，所以两边 each 地址不一样的
        each = str(each)
        max_len = max(max_len, len(each))

    nums.sort(reverse=True)

    return ''.join(nums)

if __name__ == '__main__':
    sol = Solution()
    node1 = ListNode(3)
    node2 = ListNode(1)
    node3 = ListNode(6)
    node4 = ListNode(4)
    node5 = ListNode(5)
    node6 = ListNode(2)
    node7 = ListNode(9)
    node8 = ListNode(0)
    node9 = ListNode(10)
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    node5.next = node6
    node6.next = node7
    node7.next = node8
    node8.next = node9
    l = sol.sortList_iterative(node1)
    print(l)
    pass
