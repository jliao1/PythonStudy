class Solution:

    # 力扣1 E Two Sum 双指针做法
    def twoSum(self, nums, target):
        ''' 我这个时间复杂度是 O(NlogN) '''
        numbers = [(num, i) for i, num in enumerate(nums)]
        numbers.sort(key=lambda x: x[0])

        left = 0
        right = len(nums) - 1

        while left < right:
            current_sum = numbers[left][0] + numbers[right][0]
            if current_sum == target:
                return [numbers[left][1], numbers[right][1]]
            if current_sum > target:
                right -= 1
            if current_sum < target:
                left += 1

        return [-1, -1]

    # 力扣1 E Two Sum 双指针做法 哈希表做法
    def twoSum(self, nums, target):
        ''' 我这个时间复杂度是 O(n) '''
        hash_map = {}
        for i, num in enumerate(nums):
            if (target - num) in hash_map:
                return [i, hash_map[target - num]]

            hash_map[num] = i

        return [-1, -1]

    # 力扣M 8 String to Integer (atoi) 对strng的处理，各种分情况讨论，edge case多
    def myAtoi2(self, s: str) -> int:
        '''
        有些edge case题目还是没澄清，比如
        "  +  413" 这种输出是0
        " +-43" 这种输出也是0
        我这个时间复杂度是 O(n)  但最终下来我的速度好像有点慢
        '''
        sign = None
        num = None

        for i, c in enumerate(s):

            if sign is None and num is None and c in ' ':
                continue

            if sign is None and num is None and c in '+-':
                sign = c
                continue

            if c.isdigit():
                if num is None:
                    num = 0

                num = num * 10 + int(c)

                valid, temp_num = self.is_valid8(sign, num)
                if not valid:
                    return temp_num

                continue

            break

        valid, temp_num = self.is_valid8(sign, num)
        return temp_num
    def is_valid8(self, sign, num):
        res = 0
        if num is None:
            num = 0

        if sign == '+' or sign is None:
            res = num
        if sign == '-':
            res = - num

        if -2 ** 31 <= res <= 2 ** 31 - 1:
            return True, res
        elif res > 2 ** 31 - 1:
            return False, 2 ** 31 - 1
        else:
            return False, -2 ** 31

    # 力扣M 3 Longest Substring Without Repeating Characters 经典通向双指针 Sliding Window 用字典做的
    def lengthOfLongestSubstring1(self, s):
        '''
        edge case怎么又忘了呀唉
        当输入为 '' 和 ' '

        同向双指针 Sliding Window 解法
        这是我自己写的，用双指针 + hash_map 做的
        '''
        m = {}
        max_len = 0
        start = 0

        for i, c in enumerate(s):

            if c not in m:
                length = i - start + 1
                max_len = max(max_len, length)
                m[c] = i

                continue

            else:  # c in m:
                c_i = m[c]
                # 说嘛从 start 到 c_i的都得从map里去掉
                self.remove3(start, c_i, m, s)

                start = c_i + 1
                m[c] = i
                continue

        return max_len
    def remove3(self, start, end, m, s):
        for i in range(start, end + 1):
            m.pop(s[i], None)  # 这种写法不会报错，如果key存在就移除并返回key的值，如果不存在就返回None

    # 力扣M 3 Longest Substring Without Repeating Characters 经典通向双指针 Sliding Window 用hash_set做的
    def lengthOfLongestSubstring2(self, s):
        if s == '':
            return 0

        window = set()

        left = 0

        max_len = 0

        for ch in s:

            # 当ch已存在于window中, 要把left到ch第一次出现的位置期间的元素都删掉，然后left移动到ch第二次出现的位置，怎么做呢？
            while ch in window:
                # left 删一个移一位，直到 ch不再存在 window里，此时的 left就指向 ch 第二次出现的地方了
                window.remove(s[left])
                left += 1

            # 添加ch

            window.add(ch)

            # 更新长度

            max_len = max(max_len, len(window))

        return max_len

    # 力扣 M 12 Integer to Roman 用Greedy方法
    def intToRoman(self, num):
        """
        这个解法的时间空间O(1)

        divmod(divident, divisor) 是用来取商和余数的
        例如 x = divmod(5, 2)    x是(2,1)

        # A Greedy algorithm is an algorithm that makes the best possible decision at the current time;
        # in this case taking out the largest possible symbol it can.
        """
        # 这好像是 prior knowledge，由这些一定能转出来唯一的罗马数字
        # 降序排列的
        digits = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                  (90, "XC"),   (50, "L"),  (40, "XL"), (10, "X"),   (9, "IX"),
                  (5, "V"),     (4, "IV"),  (1, "I") ]

        roman_digits = []

        # Loop through each symbol.
        for value, symbol in digits:
            # We don't want to continue looping if we're done.
            if num == 0:
                break
            # To represent a given integer, we look for the largest symbol that fits into it.
            # We subtract that, and then look for the largest symbol that fits into the remainder,
            # and so on until the remainder is 0
            count, num = divmod(num, value)
            #  Each of the symbols we take out are appended onto the output Roman Numeral string. (Append "count" copies of "symbol" to roman_digits.)
            roman_digits.append(symbol * count)

        return "".join(roman_digits)

    # 是12题的反转题，做下当练习   力扣 E 13. Roman to Integer
    def romanToInt(self, s: str) -> int:
        Map = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }

        res = 0

        i = 0
        while i < len(s):

            if i + 1 < len(s) and Map[s[i]] < Map[s[i + 1]]:
                res += Map[s[i + 1]] - Map[s[i]]
                i += 2
            else:
                res += Map[s[i]]
                i += 1

        return res

    # 力扣 11 M Container With Most Water 双指针滑动窗口greedy，唉还是有点不理解为啥这种做法就能找出 maximum
    def maxArea(self, height):
        """
        The intuition behind this approach is that the area formed between the lines will always be limited by the height of the shorter line.
        Further, the farther the lines, the more will be the area obtained.
        时间空间 O(1)
        唉还是有点不理解（1）为啥这种做法就能找出maximum
                      （2）为什么 height[left] == height[right] 时不单独处理？
        """
        # maintain a variable max_area to store the maximum area obtained till now
        max_area = 0
        # We take two pointers, one at the beginning and one at the end of the array constituting the width of 2 lines
        # we start from the exterior most lines because Initially we want the width to be as big as possible
        left = 0
        right = len(height) - 1

        while left < right:
            max_area = max(max_area, (right - left) * min(height[left], height[right]))
            # moving the pointer (pointing to the shorter line) towards the other end  by 1 step
            # might overcome the reduction in the width
            if height[left] <= height[right]:  # 为什么 height[left] == height[right] 时不单独处理？
                # keep relatively longer line (keep right)
                left += 1
            else:
                right -= 1
        return max_area

    # 力扣 15 3Sum  双指针 + 降维 + 利用排序去重
    def threeSum(self, nums) :
        """
        自己做的时候，对于该怎么去重，懵逼了。原来排序不仅有利于查找数，还有利于去重，唉
        返回时以 a <= b < = c 的pattern来返回的，一开始对a作for loop, 从a的右边范围去查找b和c，这样就可以去重了，这样也把3sum的问题降维成2sum

        比如输入是这个 [-1,-1,-1,0,1,2]  输出是 [[-1, -1, 2], [-1, 0, 1]]
        对于该返回啥不清楚。它的要求是你返回的很多个tuple里，一个tuple里的3个元素的index必须不同，tuple之间两两不同
        """
        nums.sort()
        res = []

        for i, a in enumerate(nums):
            # skip the identical element
            # 下标有效检测    当 nums[i] 等于前一个数, 那就不需要再对 nums[i] 进行处理了，对nums[i]进行找的话找出来的情况一定比num[i-1]的少
            if i - 1 >= 0 and nums[i - 1] == nums[i]:
                continue

            target = -a
            self.find_2nums_equals_to_target1(i, nums, res, target)

        return res
    # 双指针做法
    def find_2nums_equals_to_target1(self, i, nums, res, target):
        # 从 left 到 right 范围找，就避免重复查找，并且找出来的3个数字index肯定不同
        left = i + 1
        right = len(nums) - 1

        while left < right:
            sum2 = nums[left] + nums[right]

            if sum2 > target:
                right -= 1
            elif sum2 < target:
                left += 1
            else:  # sum2 = target:
                res.append([-target, nums[left], nums[right]])
                # 我们需要继续前进，知道左右指针相遇
                left += 1
                # 这一步去重，保证 tuple 之间两两不同
                while left < len(nums) and nums[left - 1] == nums[left]:
                    left += 1
    # 哈希表做法
    def find_2nums_equals_to_target2(self, i, nums, res, target):
        left = i + 1
        seen = set()

        while left < len(nums):
            temp = target - nums[left]
            if temp in seen:
                res.append([-target, nums[left], temp])
                # 要看一下相不相等，相等的话才移动left，因为后面要把 left add 进 seen里，所以 left指向的内容至少不能变
                while left + 1 < len(nums) and nums[left] == nums[left + 1]:
                    left += 1

            seen.add(nums[left])
            # 加完了后继续处理下一个
            left += 1

    # 力扣 M 16 3Sum Closest 降维+双指针  这个解法更精准
    def threeSumClosest(self, nums, target) :
        Sum = 0
        nums.sort()
        difference = float('inf')
        # a <= b <= c
        # a + b + c -> target

        for i, a in enumerate(nums):
            if i - 1 >= 0 and nums[i] == nums[i - 1]:
                continue

            left = i + 1
            right = len(nums) - 1

            while left < right:
                sum3 = nums[i] + nums[left] + nums[right]

                if abs(sum3 - target) < difference:
                    difference = abs(sum3 - target)
                    Sum = sum3

                if sum3 < target:
                    left += 1
                elif sum3 > target:
                    right -= 1
                else:
                    return sum3

        return Sum

    # 力扣 28 E Implement strStr()  字符串查找（好像有种叫 KMP的算法，不知道需要掌握不）
    def strStr(self, haystack: str, needle: str) -> int:
        """
        这种做法好像是O(m*n)的时间复杂度，因为
        """
        if not needle:
            return 0

        if not haystack:
            return -1

        if haystack.find(needle) == -1:
            return -1
        else:
            return haystack.find(needle)

        """
        手写代码是
        if source is None or target is None:
            return -1
        len_s = len(source)
        len_t = len(target)
        for i in range(len_s - len_t + 1):
            j = 0
            while (j < len_t):
                if source[i + j] != target[j]:
                    break
                j += 1
            if j == len_t:
                return i
        return -1

        """

    # 力扣 M 48. Rotate Image 除非做过不然死都整不出来，对二维数组的翻转骚操作
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.

        先找出规律 (x,y)  -->  (y, n-1-x)
        step1  (x,y)  --> (y, x)         对角线 翻转
        step2  (y,x)  --> (y, n - 1 - x) 同一行 翻转

        """
        n = len(matrix)
        # 对角线 翻转
        for x in range(n):
            for y in range(x + 1, n):
                matrix[x][y], matrix[y][x] = matrix[y][x], matrix[x][y]
        # 每一行 翻转
        for x in range(n):
            matrix[x].reverse()

    # 力扣 M 49. Group Anagrams 做法是 Categorize by Sorted String
    def groupAnagrams1(self, strs):
        m = {}
        for each in strs:
            # Categorize by Sorted String 这样做最后时间复杂度是 O(NKlogK)
            temp = sorted(each)   #  注意这里返回的是一个 list，不能直接当成哈希表的key用
            temp = ''.join(temp)  # 要转成string 或者转成 tuple再存
            if temp not in m:
                m[temp] = []
                m[temp].append(each)
                continue

            m[temp].append(each)

        return [m[each] for each in m]  # 空间复杂度O(NK)
        #也可以写成 return  m.values()

    # 力扣 M 49. Group Anagrams 做法是 Categorize by Count 时间空间都是O(NK)
    def groupAnagrams2(self, strs):
        m = {}
        for string in strs:
            # 小技巧：用了26个字母当index去存，因为这里已经说了都是小写英文字母了, 可以转成很自由26个字母
            count = [0] * 26
            for c in string:
                # 把英文小写字母转成 26位 array的下标
                count[ord(c) - ord('a')] += 1

            # 走到这一步 count 已经sort好了
            m.setdefault(tuple(count), [])
            m[tuple(count)].append(string)

        return m.values()

    # 力扣 H 76. Minimum Window Substring  同向双指针滑动窗口 + hashmap记数，里面的规律实在是太细节得难以摸清
    def minWindow1(self, s: str, t: str) -> str:
        # 这个版本代码比较少  时间复杂度貌似是O(2N+k)  为什么2N因为左右双指针移动1次
        from collections import Counter

        res = ""
        l, r, cnt, minLen = 0, 0, 0, float('inf')
        s_count = Counter()  # 反正是个计数器
        t_count = Counter(t)

        while r < len(s):
            s_count[s[r]] += 1

            # 在s中找到了t的元素，但还没找满
            if s[r] in t_count and s_count[s[r]] <= t_count[s[r]]:
                cnt += 1  # 这是一个比方法2加快速度的小技巧， 太难了

            #  当窗口还没滑完   找够了
            while l <= r and cnt == len(t):
                # 存结果
                if minLen > r - l + 1:
                    minLen = r - l + 1
                    res = s[l: r + 1]

                # s_count 里最左边弹出去一个
                s_count[s[l]] -= 1
                # 如果弹出去的这个元素在 t 里，那么 cnt 要-1
                if s[l] in t_count and s_count[s[l]] < t_count[s[l]]:
                    print(s[l])
                    cnt -= 1
                # 弹了，左指针向右移动一位
                l += 1

            r += 1
        return res

    # 力扣 H 76. Minimum Window Substring  同向双指针滑动窗口 + hashmap记数，这是自己做的版本有点慢
    def minWindow2(self, s: str, t: str) -> str:
        # 这个做法貌似时间复杂度是 O(nk) 空间是O(n+k)
        from collections import Counter

        T = Counter(t)
        S = {}

        min_len = float('inf')
        result = ''

        l = 0
        r = 0

        while r < len(s) and l <= r:

            c = s[r]

            if c in T and c not in S:
                S[c] = 1
            elif c in T and c in S:
                S[c] += 1

            while self.is_valid(T, S):

                current_len = r - l + 1
                if current_len < min_len:
                    min_len = current_len
                    result = s[l:r + 1]

                # pop S[l]
                if s[l] in S:
                    S[s[l]] -= 1  #
                    if S[s[l]] == 0:
                        S.pop(s[l])

                l += 1

            r += 1

        return result
    def is_valid(self, T, S):  # O(k)
        for key in T:
            if key in S and T[key] <= S[key]:
                continue
            else:
                return False

        return True

    # 力扣 M 165. Compare Version Numbers 字符串的操作
    def compareVersion(self, s, t):

        s = [int(element) for element in s.split('.')]
        t = [int(element) for element in t.split('.')]

        # 这种写法比较简略
        for i in range(max(len(s), len(t))):
            v1 = s[i] if i < len(s) else 0
            v2 = t[i] if i < len(t) else 0

            if v1 > v2:
                return 1
            if v1 < v2:
                return -1

        return 0

    # 力扣 M 238. Product of Array Except Self 前缀和的思想！！这里是前缀积，后缀积，找出这种规律也蛮难的！
    def productExceptSelf(self, nums):
        ans = [1] * len(nums)
        prefix_product = 1
        suffix_product = 1

        for i in range(len(nums)):
            ans[i] = prefix_product
            # update prefix_product
            prefix_product = prefix_product * nums[i]

        for i in range(len(nums) - 1, -1,-1):
            ans[i] = ans[i] * suffix_product
            suffix_product = suffix_product * nums[i]

        return ans

    # 力扣 M 268. Missing Number  利用等差数列求和公式
    def missingNumber1(self, nums):
        '''
        这是利用数学公式，有点像抖了个机灵的方法
        但时间的确是O(n) 空间是O(1)

        除此之外还可以用sort一下nums的方法，时间就变成O(NlogN)
        除此之外还可以先hashmap，空间就变成O(N)
        看方法2，有点难想到但是挺聪明
        '''
        # 不用考虑 integer overflow因为Integers are are implemented as “long” integer objects of arbitrary size in python3 and do not normally overflow.
        expected_sum = len(nums) * (len(nums) + 1) // 2
        actual_sum = sum(nums)  # O(n)
        return expected_sum - actual_sum

    # 力扣 M 268. Missing Number  利用array角标，内部排序， 太 tricky了吧！！
    def missingNumber2(self, nums):
        """时O(n) 空O(1)"""
        n = len(nums)
        i = 0

        # 这个技巧太tricky了吧…真特么想不到
        while i < n:
            # while 条件
            #(1)还没走到它该到的位置上 (2)不是最后一个数 nums[i] == n的情况
            while nums[i] != i and nums[i] < n:
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]
            i += 1

        for k in range(n):
            if nums[k] != k:
                return k

        return n

    # 力扣 H 273 Integer to English Words
    def numberToWords(self, num):
        if num == 0: return 'Zero'

        d = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
            10: "Ten",
            11: "Eleven",
            12: "Twelve",
            13: "Thirteen",
            14: "Fourteen",
            15: "Fifteen",
            16: "Sixteen",
            17: "Seventeen",
            18: "Eighteen",
            19: "Nineteen",
            20: "Twenty",
            30: "Thirty",
            40: "Forty",
            50: "Fifty",
            60: "Sixty",
            70: "Seventy",
            80: "Eighty",
            90: "Ninety",
            100: "Hundred",
            1000: "Thousand",
            1000000: "Million",
            1000000000: "Billion"  # 其实最大是  2**31 - 1 = 2 147 483 647
        }
        # helper 处理 1 - 1000 的情况
        def helper(i):
            if i <= 20:
                return d[i]
            elif i < 100:
                #      不是整十的话，返回 几十 + 多少         判断是否 整十       整十的话就只返回 几十
                return d[i // 10 * 10] + ' ' + d[i % 10] if i % 10 > 0 else d[i // 10 * 10]
            elif i < 1000:
                #      不是整百的话，返回 几百 + 多少                    判断是否 整百         整百的话就只返回 几百
                return d[i // 100] + ' Hundred ' + helper(i % 100) if i % 100 > 0 else d[i // 100] + ' Hundred'

        res = []

        if num >= 1000000000:
            res.append(d[num // 1000000000])  # 这里也可以写成 res.append( helper(num//1000000000) ) 但没必要了 因为最大 2**31-1 是 2 billion
            res.append( d[1000000000] )
            num = num % 1000000000

        if num >= 1000000:
            res.append(helper(num // 1000000))
            res.append( d[1000000] )
            num = num % 1000000

        if num >= 1000:
            res.append(helper(num // 1000))
            res.append( d[1000] )
            num = num % 1000

        if num > 0:
            res.append(helper(num))

        return ' '.join(res)

    # 力扣 E 387. First Unique Character in a String
    def firstUniqChar(self, s: str) -> int:
        """
        时间复杂度O(n)
        空间复杂度O(1) 虽然咱们这里用了hashmap但一共也就26个字母所以空间是常数
        """
        from collections import OrderedDict
        # 队列字典
        d = OrderedDict()
        for c in s:
            if c not in d:
                d[c] = 1
            else:
                d[c] += 1

        while d:
            key, value = d.popitem(last=False)  # 这样才是以队列的形式先进先pop出来的
            if value == 1:
                non_repeating_character = key
                return s.find(non_repeating_character)

        return -1

    # 力扣 E 819. Most Common Word  string+hashmap+排序骚操作，edge case不少的，对我来说难度是 Medium
    def mostCommonWord(self, paragraph, banned):
        """
        时间/空间 复杂度O(M+N)
        N be the number of characters in the input string, M be the number of characters in the banned list.
        """
        counter = {}  # 空间O(N)
        word = []
        Set = set(banned)  # 空间O(M)，时间也是O(M)??
        Set.add('')  # 为了不统计 空str

        for i, c in enumerate(paragraph.lower()):  # 时间O(N)
            if c.isalnum() and i != len(paragraph) - 1:
                word.append(c)
            else:
                # 处理最后一个char
                if i == len(paragraph) - 1 and c.isalnum():
                    word.append(c)

                word = ''.join(word)
                if word not in Set:
                    counter[word] = counter.get(word, 0) + 1  # 这句也很漂亮

                word = []

        # 以下两句是教如何在 map_couter中找出最大value的key
        List = [(key, value) for key, value in counter.items()]
        return max(List, key=lambda item: item[1])[0]


if __name__ == '__main__':
    input1 = "Bob. hIt, baLl"
    input2 = ["bob", "hit"]
    sol = Solution()
    res = sol.mostCommonWord(input1, input2)

    print(res)