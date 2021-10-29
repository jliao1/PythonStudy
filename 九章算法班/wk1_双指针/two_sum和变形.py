"""
wk1_双指针(竟然是可以避免重复的)
哈希表
partition
降维，折半查找
批量统计
"""
class Solution:
    """【第一大类(2sum找某值) 找到一个解就返回的 不用处理 duplicate元素 】"""
    # lintcode E 608 Two Sum某值- 双指针做法更好，因为数组已sort，返回index组(低index在前)，因为solution只有一组(不用考虑duplicate number)
    def twoSum(self, nums, target):
        l, r = 0, len(nums)-1
        while l < r:
            value = nums[l] + nums[r]
            if value == target:     # 因为只有一个解所以好像可以直接返回，不用考虑duplicate number
                return [l+1, r+1]   # 返回的 index1 must be less than index2
            elif value < target:
                l += 1
            else:
                r -= 1
        return []

    # lintcode E 56 Two Sum某值 双指针做法  数组未排序，返回一个index组(低index在前)，因为solution只有一组(不用考虑处理duplicat元素)
    def twoSum1(self, nums, target):
        '''
        这是双指针做法
        解题思路是先从小-》大sort一遍
        最left是最小，最right是最大
        看 nums[left] + nums[right] 和 target 是？关系
                                        大于的话，排除掉最大的数
                                        小于的话，排除掉最小的数
                                        等于，就找到答案了
        【双指针和哈希表做法对比】
        如果要求返回数组里的具体数字就好，
        或 如果输入数据是已经排序好了的（方便返回index），就是领扣608题
        那双指针的做法时间复杂度就是O(n), 空间是O(1)  这比哈希表做法好

        但这道题输入的nums是无序的，要求返回的是 index，
        所以双指针做法，先排序要 NlogN，空间也要用到 O(n) 来存 index。
        而哈希表做法时间空间都是O(n), 哈希表更好
        '''
        if len(nums) < 2:
            return [-1, -1]

        temp = [ (num, i) for i, num in enumerate(nums)]

        temp.sort(key=lambda n: n[0])

        left = 0
        right = len(nums) - 1
        while left < right:
            if temp[left][0] + temp[right][0] < target:
                left += 1
            elif temp[left][0] + temp[right][0] > target:
                right -= 1
            else:
                return temp[left][1], temp[right][1] #  index1 must be less than index2

        return [-1, -1]

    # lintcode E 56 Two Sum某值 哈希表做法更快  数组未排序，返回一个index组(低index在前)，因为solution只有一组(不用考虑duplicate number)
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

    """【第二大类(比大小/计数问题) 统计数量，允许组里有重复元素的，只是每个index用一次就好，那就不用考虑处理duplicate元素 】"""
    # lintcode M 609 Two Sum某值比大/小计数问题Less than or equal to target  数组未sort，返回满足条件组的个数。这种先sort，用双指针做，可以加速by批量处理
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
                # 加速by批量处理(同时把duplicate number情况也考虑进去处理了)
                count += r - l
                # 左指针继续往中间走，直到左右指针相遇就停下。这样也可以去重，出来的结果不会说相同位置的数字进行重复统计
                l += 1
        return count

    # lintcode M 443 Two Sum某值比大小/计数问题Greater than target  数组未sort，返回满足条件组的个数。这种先sort，用双指针做，可以加速by批量处理
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

    # lintcode M 533 Two Sum某值比大小问题Closest to target   数组未sort，返回唯一(最近)的解，找这种答案不用过滤duplicate的解也没事 跟前几题有点不一样，思路差不多。
    def twoSumClosest(self, nums, target):
        """
        逻辑上和基础的2 sum一样，反正就是一直往 target 靠。 小了就left指针右移，大了就right指针左移
        在往 target 靠的过程中，
        注意每次更新 closest 的时候取较小者
        这一路下来就能找到最小
        """
        nums.sort()

        left = 0
        right = len(nums)-1

        closest = float('inf')

        while left < right:  # 易错点，这里不能相等
            temp = nums[left] + nums[right]
            closest = min(closest, abs(temp-target))
            if temp < target:
                left += 1
            if target < temp:
                right -= 1
            if temp == target:
                break
        return closest

    # lintcode M 382 Triangle Count for降维 + 某值比大小/计数变形 返回满足条件组的个数（这题元素值的组合不用unique的，但index组合必须是unique的，所以input排序后进行依次for处理）
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
        # 这个还是要把S排序一下，从大->小依次处理，因为出来的组合的index必须是unique的，相同元素不重复处理！
        for i in range(len(S) - 1, 1, -1):  # S[i] for的是题目分析里的 c
            left = 0
            right = i - 1
            while left < right:
                two_sum = S[left] + S[right]
                if two_sum > S[i]:  # 说明可以组成三角形
                    # 批量处理一下，组合里元素值相同没事，不用处理
                    count = count + right - left
                    # 易错点 right 要继续走，因为后面还可能会有
                    right -= 1
                if two_sum <= S[i]:
                    left += 1
        return count

    # lintcode medium 976 · 4Sum II某值计数问题  哈希表折半降维查询(把4sum变2sum)  return组合里元素值可以重复，但元素的index不能重复(一个index的元素只能用一次)
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

    # lintcode medium 976 · 4Sum II 思路与方法1一样，只是代码简化了一丢丢
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

    """【第三大类(简单) 统计数量/返回所有元素组合，要求组合元素值必须是unique的。这种就必须处理duplicate元素了 by 排序从小到大依次处理+遇到已处理过的相同元素值就skip！！！】"""
    # lintcode M 57 3Sum for降纬1次 + 双指针做法  数组未sort，返回所有满足条件的数字组合，因为solution里有多组(要考虑对duplicate number的特殊处理 by 排序从小到达处理+遇到已处理过的相同元素就skip)
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

        可以用哈希表来做, 总时间复杂度 O(n^2)，空间复杂度O(n)
        但由于我们在进入two_sum_equal_to() 这个函数时，数组已经sort好了，那就用双指针来做的更快，总时间复杂度 O(n^2)，空间复杂度O(1)
        """
        self.res = []
        # 先排序，要 return 的3个数字是要找 a ≤ b ≤ c的，对a进行for循环，后面找two sums也是要基于对于排序数组的操作，方便去重
        # ！！！这种先排序可以去（1）去重相同答案（2）加快速度
        nums.sort()   # 去重技巧1：排序从小到大依次处理，避免重复处理
        # 开始降维了，由于a + b + c = 0, 下面这个 for 是对 a = nums[i] 来for的, 这样就把题目降成二维了
        for i in range(len(nums)):
            # 下标有效检测
            # 去重技巧2：若这个元素已等于前一个，不再重复处理，不然会被加到result里
            if (i-1) >= 0 and nums[i] == nums[i-1]:  # 要养成习惯，当对
                continue
            # 然后开始找，-a = b + c
            target = - nums[i]
            self.two_sum_equal_to(target, i, nums)
        return self.res
    def two_sum_equal_to(self, target, i, nums):
        # 进到这里，nums已经是sort好了的，就用双指针来做空间消耗最小(不用哈希表了)
        # 由于two sum 也就是 b + c 的和是 target = -a = -nums[i]
        # 是在 i+1 ~ len(nums)-1 范围内找的（就避免重复查找，并且找出来的3个数字index肯定不同）
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

                '''
                去重：这个是 left 移1步后，发现跟前一个相等，就为了避免重复，移到直到nums[left]跟前一数不等的时候
                      比如会有这种情况 -47 1 1 2 45 46 46
                                        l            r  
                left 和 right 移动一位后:   l        r  又找到一组1+46
                                                      为了避免重复加一个while循环 去掉这层重复
                '''
                left += 1
                # 去重点3：相同元素skip，不再重复处理，不然会被加到result里
                while (left < right) and nums[left] == nums[left-1]: # 易错点：如果这个数和前一个相等，skip，直到跟前一个数不等
                    left += 1

    # lingcode M 58 · 4Sum for降维2次 + 双指针做法  跟领扣57一毛一样思想
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


class TwoDifference:
    """【第四大类(2difference两数之差)】"""
    # 领扣 1796 K-Difference 数组无序并且元素distinct，用哈希表做最快O(n)
    def KDifference(self, nums, t):  # nums里的元素无重复
        """
        num1           num2            set     count
         1          3 or -1            1        0
         3          5     1            3        1
         5          7     3            5        2
         7          9     5            7        3
        """
        m = set()
        count = 0

        for num1 in nums:
            # 这里要注意 num2 可能有 2 个 possible 解！！
            num2 = num1 + t
            if num2 in m:
                count += 1

            num2 = num1 - t
            if num2 in m:
                count += 1

            m.add(num1)

        return count

    # 推特OA题 力扣532 K-diff Pairs 数组无序但有duplicate的，哈希表直接做更快 但对duplicate的k=0时候的处理就比较绕了 这版是自己写的，看方法2，哈希表更简洁写法
    def findPairs1(self, nums, k):
        """时间空间O(n) 更简洁看写法2"""
        if k == 0:
            from collections import Counter
            counter = Counter(nums)
            count = 0
            for value in counter.values():
                if value >= 2:
                    count += 1
            return count

        # 接下来是 k ！=0 的情况
        Set = set()
        count = 0
        for num in nums:
            if num in Set:
                continue

            # 这里 num-k 和 num+k 都不能少。因为这个哈希表做法是，看一个，加一个数进去
            if num - k in Set:
                count += 1

            if num + k in Set:
                count += 1

            Set.add(num)

        return count

    # 推特OA题 力扣532 K-diff Pairs 这题很特别妙啊！ 要求pairs值是unique的(要考虑对duplicate元素的特殊处理)
    def findPairs2(self, nums, k):

        result = 0
        from collections import Counter
        counter = Counter(nums)

        for x in counter:
            # 妙1，k大于0说明 肯定是找不同值的元素
            if k > 0 and x + k in counter: # 妙2 这种写法只找 x+k 就好
                result += 1
            # 下两句话要省掉，因为比如k=2，看1的时候会找到（1,3）看3的时候又会找到（3，1）就重复了
            # if k > 0 and x - k in counter:
            #     result += 1

            # 妙3  若k==0 要对duplicate元素的特殊处理
            if k == 0 and counter[x] > 1:
                result += 1
        return result

    # 推特OA题 力扣532 K-diff Pairs 我自己写的双指针做法，两数之差竟然是用同向！双指针！！  由于有duplicate元素，就要用到滑动去重
    def findPairs(self, nums, k):
        nums.sort()

        r = len(nums) - 1  # right pointer
        l = r - 1
        counter = 0
        # 1 1 3 4 5  k=2
        #       lr
        while l < r and l >= 0:
            temp = nums[r] - nums[l]
            if temp < k:
                l -= 1
            elif temp > k:
                r -= 1
                # 当right<-位后遇到left，就总再把left移一位
                if l >= r and l - 1 >= 0:
                    l -= 1
            else:  # nums[l] - nums[r] == k
                counter += 1

                r -= 1
                while r >= 0 and nums[r] == nums[r + 1]:
                    r -= 1
                # 当right<-位后遇到left，就总再把left移一位
                if l >= r:
                    l = r - 1

        return counter


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
    """wk1_双指针, 但add是O(n)所以超时了啦。但这也是一种思路"""
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

    # wk1_双指针, 基于排序数组做是O(n)
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

class Ladder:

    # 领扣M 587 · Two Sum - Unique pairs  双指针做 时间复杂度O(NlogN)
    def twoSum61(self, nums, target):
        nums.sort()
        left = 0
        right = len(nums)-1
        s = set()

        while left < right:
            a = nums[left]
            b = nums[right]

            if a+b == target:
                s.add((a,b))
                left += 1
            if a+b > target:
                right -= 1
            if a+b < target:
                left += 1

        return len(s)

    # 领扣M 587 · Two Sum - Unique pairs 哈希表来记录哪些查没查过  时间复杂度O(N)
    def twoSum62(self, nums, target):
        """
        使用hashMap记录array中元素的使用情况，
        未使用为0
        使用过的为1
        不去重，不排序，时间复杂度O(n)，但空间要耗占一些
        """
        if len(nums) <= 1:
            return 0
        dic = {}
        num = 0
        for num in nums:
            temp = target - num
            # 如果temp在dic里，并且 dic[temp] 没被使用过
            if temp in dic and dic[temp] == 0:
                # 配对了
                num += 1
                # temp 标记为使用
                dic[temp] = 1
                # num 标记为使用
                dic[num] = 1
            if num not in dic:
                dic[num] = 0
        return num


if __name__ == '__main__':
    n = [3,2,4]


    sol = Solution()
    l = sol.twoSumClosest([-1,2,1,-4],4)
    pass