"""
双指针(竟然是可以避免重复的)
哈希表
partition
降维，折半查找
批量统计
"""
class Solution:

    # lintcode easy 56 · Two Sum 双指针做法
    def twoSum1(self, nums, target):
        '''
        这是双指针做法
        解题思路是先从小-》大sort一遍
        最left是最小，最right是最大
        看 nums[left] + nums[right] 和 target 是？关系
                                        大于的话，排除掉最大的数
                                        小于的话，排除掉最小的数
                                        等于，就找到答案了

        如果要求返回数组里的具体数字就好，如果输入数据是已经排序好了的
        那双指针的做法时间复杂度是O(n), 空间是O(1)  比哈希表做法好

        但这道题要求返回的是 index，
        所以双指针做法，排序要 NlogN，空间也要用到 O(n) 来存 index。
        而哈希表做法时间空间都是O(n), 哈希表更好
        '''
        if len(nums) < 2:
            return [-1, -1]

        temp = [
            (num, i)
            for i, num in enumerate(nums)
        ]

        temp.sort(key=lambda n: n[0])

        left = 0
        right = len(nums) - 1
        while left < right:
            if temp[left][0] + temp[right][0] < target:
                left += 1
            elif temp[left][0] + temp[right][0] > target:
                right -= 1
            else:
                return temp[left][1], temp[right][1]

        return [-1, -1]

    # lintcode easy 56 · Two Sum 哈希表做法
    def twoSum2(self, numbers, target):
        '''哈希表做法，时间空间都是O(n)'''
        dict = {}
        for i, val in enumerate(numbers):
            if target - i in dict:
                '''
                能执行返回语句的话，
                先找到的一定是下标比较小的那个，
                因为我们是按下标遍历过来的
                '''
                return dict[target - i], i
            '''
            要先检测完了，再加到字典里
            因为如果输入是这个 [2,4,5] target = 8
            会返回 1,1 的
            '''
            dict[val] = i

    # lintcode Medium 609 · Two Sum - Less than or equal to target 批量
    def twoSum5(self, nums, target):
        ''' 空间复杂度O(n) '''
        l, r = 0, len(nums)-1
        count = 0
        nums.sort()
        while l < r:
            sum = nums[l] + nums[r]
            if sum > target:
                r -= 1
            else:
                # 批量处理
                count += r - l
                # 指针继续往中间走，直到左右指针相遇就停下。这样也可以去重，出来的结果不会说相同位置的数字进行重复计算
                l += 1
        return count

    # lintcode Medium 443 · Two Sum - Greater than target 批量处理
    def twoSum2(self, nums, target):
        nums.sort()
        left = 0
        right = len(nums)-1
        count = 0
        while left < right:
            temp_sum = nums[left] + nums[right]
            if temp_sum > target:
                count = count + right - left
                right -= 1
            if temp_sum <= target:
                left += 1

        return count

# lintcode easy 607 · Two Sum III - Data structure design 哈希表做法
class TwoSum1:
    """
    拿到这个题，要问，这两个方法，
    被调用频次是一样多，还是谁更多？
    这会决定用什么方法实现，来踩中面试官好想要的答案
    其实本质上还是要知道自己设计的数据结构主要能干啥，主要用来实现什么的

    如果 add调用得很多，find调用得很少。
    那用哈希表的方法做就很不错，就是内部维护一个hash。
    这样add的时间复杂度就是O(1), find就是O(n)
    那用set还是map呢？
    要用map，因为map可以记录比如一个数字出现次数，比如4出现了2次，
    我们就知道它可以组成8. 用set的话两个4就会被去重成一个4就不造了

    除了哈希表算法之外，探讨其他做法
    add函数里，想要一边add一边保持数组有序，
    find也需要基于一个有序的数组才能O(n), 该怎么弄？
    (1) binary search + array insert？
        不行，binary search会消耗logN，但因为在array里插入一个东西，后移其他元素也要花费O(n)
    (2) binary search + linked list
        不行因为 binary search 是基于数组的算法，不是基于链表的
        虽然在链表中插入一个是O(1), 但找到它是O(n)
        无法通过角标花O(1)的时间去list里查找到这个数
    (3) heap 也不行
        heap 取最大值可以是 O(1)
        想要知道第二大 value，必须先删掉第一个 value。因为堆是一个树状结构，堆内部元素组织顺序不是有序的
        在堆当中，插入虽然是 logN
        但从堆里从小到大拿出 n 个元素，需要 O(NlogN) 的时间 （这部分没听懂。是令狐冲第一堂直播课38分钟，没听到）
        而且find方法需要基于一个排序数组去做，堆是无法完成的，堆内部元组组织无序的
    (4) tree map 红黑树（就是一个平衡排序二叉树，它的中序遍历是个有序数组，左右孩子高度差不超过1）
        用它来实现的 add 的话是 logN
        实现find的话就先做一次中序遍历让它变成有序的，耗时O(n), 也是可以的
        但这种方法跟用哈希表比起来，都要消耗空间，但是add是logN，而用哈希表实现add才O(1)
        而且哈希表实现起来也简单一些
    """

    def __init__(self):
        self.dic = {}

    # O(1)
    def add(self, num):
        if num not in self.dic:
            self.dic[num] = 1
        else:
            self.dic[num] += 1

    # O(n)
    def find(self, value):
        for key, count in self.dic.items():
            goal = value - key
            # 如果这个goal等于key，key出现了2次以上，就可以组成 value
            if goal == key and count >= 2:
                return True
            # 如果goal不等于key，但goal在dic里
            if goal != key and goal in self.dic:
                return True

        return False

# lintcode easy 607 · Two Sum III - Data structure design 双指针做法
class TwoSum2:
    """双指针, 但add是O(n)所以超时了啦。但这也是一种思路"""
    def __init__(self):
        self.nums = []

    # 单指针，可以使用基于排序数组的insertion sort耗时O(n)
    def add(self, num):
        self.nums.append(num)
        index = len(self.nums) - 1
        while index - 1 >= 0:
            if self.nums[index - 1] > self.nums[index]:
                self.nums[index - 1], self.nums[index] = self.nums[index], self.nums[index - 1]
                index -= 1

    # 双指针, 基于排序数组做是O(n)
    def find(self, value):
        left, right = 0, len(self.nums) - 1
        while left > right:
            temp = self.nums[left] + self.nums[right]
            if temp > value:
                right -= 1
            if temp < value:
                left += 1
            if temp == value:
                return True

        return False

class ThreeSum:

    # lintcode Medium 57 · 3Sum 降纬
    def threeSum(self, nums):
        """
        【降纬思想】
        要找 a + b + c = 0
        实际上是找 a + b = -c
        把其中一个因子for循环一下，剩下的就是2个纬度了
        因此看到4数之和也可以先降成3维

        其实2sums用哈希表做就是这种思路，
        在限定下找另一个值
        比如找 a+b = target，我们就是
        for a in hash:
            找(target-a)是否在哈希表里

        可以用哈希表来做, 时间复杂度 O(n^2)，空间复杂度O(n)
        这里是用双指针来做的，时间复杂度 O(n^2)，空间复杂度O(1)
        """
        self.res = []
        # 先排序，要 return 的3个数字是要找 a ≤ b ≤ c的，后面找two sums也是要基于对于排序数组的操作
        nums.sort()
        # 开始降维了，由于a + b + c = 0, 下面这个 for 是对 a = nums[i] 来for的, 这样就把题目降成二维了
        for i in range(len(nums)):
            # 下标有效检测    当 nums[i] 等于前一个数, 那就不需要再对 nums[i] 进行处理了
            if (i-1) >= 0 and nums[i] == nums[i-1]:
                continue
            # 然后开始找，-a = b + c
            target = - nums[i]
            self.two_sum_equal_to(target, i, nums)
        return self.res
    def two_sum_equal_to(self, target, i, nums):
        # 由于two sum 也就是 b + c 的和是 target = -a = -nums[i]
        # 是在 i+1 ~ len(nums)-1 范围内找的（就避免重复查找）
        left = i + 1
        right = len(nums) - 1
        while left < right:
            two_sum = nums[left] + nums[right]
            if two_sum < target:
                left += 1
            elif two_sum > target:
                right -= 1
            else:  # two_sum == target:
                self.res.append([nums[i], nums[left], nums[right]])
                # 找到满足条件的，并不马上退出，因为还要继续往中间找直到 left=right
                left += 1
                right -= 1
                '''
                去重：这个是 left 移1步后，发现跟前一个相等，就为了避免重复，移到直到nums[left]跟前一数不等的时候
                      比如会有这种情况 -47 1 1 2 45 46 46
                                        l            r  
                left 和 right 移动一位后:   l        r  又找到一组1+46
                                                      为了避免重复加一个while循环 去掉这层重复
                '''
                while (left < right) and nums[left] == nums[left-1]:
                    left += 1

    # lingcode Medium 58 · 4Sum 降维 + 批量(求个数)
    def fourSum(self, nums, target):
        """ 时间复杂度 O(n^3) """
        nums.sort()
        self.res = []
        # a + b + c + d = target 先对 a 进行循环 降纬
        for i in range(len(nums)-1):  # 一层O(n)
            # 检测下标有效，如果数字相同就不用找了
            if (i-1) >= 0 and nums[i] == nums[i-1]:
                continue
            # 如果是ok的，再降纬
            for j in range(i+1, len(nums)-1):  # 一层O(n)
                if (j-1) >= (i+1) and nums[j] == nums[j-1]:
                    continue
                goal = target - nums[i] - nums[j]
                self.find_two_sum(i, j, nums, goal)  # 一层O(n)

        return self.res
    def find_two_sum(self, i, j, nums, goal):
        left = j + 1
        right = len(nums) - 1
        while left < right:
            temp = nums[left] + nums[right]
            if temp > goal:
                right -= 1
            elif temp < goal:
                left += 1
            else: # temp == goal:
                self.res.append((nums[i], nums[j], nums[left], nums[right]))
                left += 1
                right -= 1

                while (left < right) and nums[left] == nums[left-1]:
                    left += 1

    # lintcode Medium 382 · Triangle Count 降维 + 批量(求个数)
    def triangleCount(self, S):
        """
        three sum 的变形
        组成三角形条件是，假设 a ≤ b ≤ c 的情况下
        两边之和大于第三边 a + b > c (这是充要条件）
        因此降纬的话，第一层 for 是按 c 来循环

        如果一个一个数的话，组合一下就是 C(3)(n) = O(n^3)级别的，肯定会超时的

        小技巧：
        通常让你求总数的题，都是【批量题】。本题解法 O(n^2)
        降纬后的处理有点类似于 领扣609 · Two Sum - Less than or equal to target

        follow up1:
        如果这题要你求出具体有哪些pairs，那只有O(n^3)
        假如要去重该怎么做？也只能暴力地做了，基本要把每个方法找到的
        总结：求具体的方案往往比较暴力一些  求方案的个数往往可以批量解

        follow up2：
        为什么three sum题目如果求具体方案可以做到O(n^2)
        而本题如果求具体方案只可以做到O(n^3)?
        因为 three sum那道题虽然是找 a + b + c = 0，但a和b确定后，c=-(a+b)就确定了，所以浮动的其实只有a和b
        但这道题 a+b > c，a和b确定后，c依然是有n种可能性的
        """
        S.sort()
        count = 0
        # S[i] for的是题目分析里的 c
        for i in range(len(S) - 1, 1, -1):
            left = 0
            right = i - 1
            while left < right:
                two_sum = S[left] + S[right]
                if two_sum > S[i]:  # 说明可以组成三角形
                    # 批量处理一下
                    count = count + right - left
                    # right 要继续走，因为后面还可能会有
                    right -= 1
                if two_sum <= S[i]:
                    left += 1
        return count

    # lintcode medium 976 · 4Sum II 折半查询 + 哈希表 + 批量
    def fourSumCount1(self, A, B, C, D):
        """
        【折半查询】
        规模从4个分成2组，
        第一组，把a和b的二元组合都找到放进哈希表里
        在第二组里，for循环c+d，看看-(c+d)是不是在哈希表里，如果在有几个数
        而这题是找四元组个数，又是一个统计方案个数的问题，利用此进行批量统计的优化
        比如前面知道 a+b=5有10种方案，后面c+d找到等于-5，我们就直接累加前面的10种方案

        时间复杂度O(n^2)
        """
        dic = {}
        length = len(A)
        for i in range(length):
            for j in range(length):
                temp = A[i]+B[j]
                if temp not in dic:
                    dic[temp] = 1
                else:
                    dic[temp] += 1
        count = 0
        for i in range(length):
            for j in range(length):
                temp = C[i]+D[j]
                if -temp in dic:
                    # 批量统计开始
                    count = count + dic[-temp]

        return count

    # lintcode medium 976 · 4Sum II 思路与1一样，只是代码简化了一丢丢
    def fourSumCount2(self, A, B, C, D):
        counter = {}
        res = 0
        for a in A:
            for b in B:
                counter[a+b] = counter.get(a+b, 0) + 1
        for c in C:
            for d in D:
                res = res + counter.get(-c-d, 0)

        return res

if __name__ == '__main__':
    n = [2, 11, 7, 15]
    sol = ThreeSum()
    l = sol.fourSum([1,0,-1,-1,-1,-1,0,1,1,1,2], 2)
    pass