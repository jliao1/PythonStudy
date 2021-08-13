# 字典讲解
'''
字典：
应用：插入和查找，平均操作效率很高O(1)
     分类记数 （其实一般可以用 collections.Counter(iterable)）
     存储且快速找到key所对应的value
     前缀之和
     双向记录构建map，单向查找set；单向记录，双向查找

非重复的无需 key-value pairs
用下标取value要先判断，key在不在
如果不在，就会报错
所以一般还是用get取，get(key, default_value) 如果找不到key就会返回default value

这种方法更常用


可以用来记录元素是不是出现过，同时还可以附带一个value值

python中的set和dict是使用hash table实现的，所以要求set元素和dict的key都是可以hash的
set的元素 和 dic的key 必须是可以 hashde
list不可以hash，如果想放一组值，转成 tuple
set不可以hash，frozenset可以
dict不可以hash
因为可变的往往不能hash

哈嘻表是典型的，时间换空间

平衡BST查找是logN，但空间消耗小，但是BST支持有序数据

但负载因子大于0.5时候，哈嘻表查找性能急剧下降。为了保证空间很大，就可以rehash，哈嘻表扩展1倍。当装的元素很少的时候，也可以rehash把空间减小
'''
import collections


class Solution:

    # lintcode(力扣246) Easy 644 · Strobogrammatic Number 分类计数
    def countCharacters(self, str):
        dic = {}
        for each in str:
            if each in dic:
                dic[each] += 1
            else:
                dic[each] = 1

        return dic

    # lintcode Easy 138 · Subarray Sum 简单的前缀和
    def subarraySum(self, nums):
        # write your code here
        if not nums:
            return [-1, -1]

        dic = {0: -1}
        Sum = 0
        for i, num in enumerate(nums):
            Sum = Sum + num
            target = Sum - 0
            if target in dic:
                return [dic[target] + 1, i]

            dic[Sum] = i

        return [-1, -1]

    # lintcode(力扣246) Easy 1789 · Distinguish Username
    def DistinguishUsername(self, names):
        hash_map = {}
        res = []
        # construct map
        for i, name in enumerate(names):
            if name not in hash_map:
                res.append(name)
                hash_map[name] = 1
            else:
                res.append(name + str(hash_map[name]))
                hash_map[name] += 1

        return res

    # lintcode(力扣1) Easy 56 · Two Sum
    def twoSum(self, numbers, target):
        """
        时间复杂度O(n) 好的情况下很快就找到，都不需要O(n)
        索引主要还是靠key去索引的
        """
        dic = {}
        for i, val in enumerate(numbers):
            if target - val in dic:  # 这个找其实是往前寻找的
                return [dic[target - val], i]
            dic[val] = i
        #无解的情况
        return -1, -1

    # lintcode medium 1457 · Search Subarray
    def searchSubarray(self, arr, k):
        """
        为什么能想到前缀和？因为有几个关键词哈，continuous，sum，sub_string
        暴力方法是，第一个 for 循环里 start指针里套一个for循环找end指针，里面再套一个for循环求start～end的和，但这种都O(n^3)
        前缀之和的方法时间复杂度O(n)
        """
        # initializing
        index = -1
        prefix_sum = 0
        dic = {prefix_sum: index} # 为什么要初始化成 0：-1   -1是用来处理edge case的，比如第一个就是k的情况

        for curr in arr:
            index += 1
            prefix_sum = curr + prefix_sum

            target = prefix_sum - k  # 因为 prefix_sum - target = k
            if target in dic:   # 如果存在的话，就直接返回，因为这题是返回the minimum ending position，所以再往后不用看的其实
                return index - dic[target]

            if prefix_sum not in dic:    # 如果存在的话，也不更新，因为这道题是返回 minimum starting position
                dic[prefix_sum] = index  # 记住第一个出现前缀和的idx就好，以后不更新了

        return -1  # 没找到返回-1

    # lintcode medium 1035 · Rabbits in Forest
    def numRabbits(self, answers):
        """
        这题需要：题目要分析一下总结规律 + 字典分类统计应用
        分析：
        如果一个兔子告诉你n，一共有多少只兔子跟这只兔子同色呢？n+1，包括这只兔子
        如果两只兔子告诉你的数字不一样，这两只兔子可能是同一个颜色嘛？ 不可能
        如果两只兔子告诉你的数字一样，这两只兔子一定是同一个颜色嘛？不一定

        一个数字n，代表这家庭最多有n+1个成员
        一个数字n最多可以出现n+1次（因为这家的，每个家庭成员出现报1次）
        不同数字，肯定是 不同的家庭，所以可以用dict

        """
        dic = dict()
        for each in answers:
            if each not in dic:
                dic[each] = 1
            else:
                dic[each] += 1

        counter = 0
        for k, v in dic.items():
            # groups是在k相同的情况下(就是兔子上报相同数字的情况下)，最多有几组不同颜色的
            groups = v // (k + 1)  # for every group,
            if v % (k + 1) != 0:
                groups += 1

            counter = counter + groups * (k + 1)

        return counter

    # lintcode(力扣246) Easy 644 · Strobogrammatic Number
    def isStrobogrammatic(self, num):
        dic = {'0':'0', '1':'1', '6':'9', '9':'6', '8':'8'}

        left = 0
        right = len(num)-1

        while left <= right:
            v1 = num[left]
            v2 = num[right]

            value = dic.get(v1, None)

            '''
            注意！！！这里一定要写成 if value is not None
            不能写成 if not value
            因为 这里value是有可能为0的，如果 value 是 0，if not value就是真
            而我们这里 value是0是允许的，所以要写成 value只要不是None就可以！！！
            '''
            if value is not None \
            and value == v2:
                left += 1
                right -= 1
            else:
                return False
        '''
        简短一点的写法2：
        map = {'0':'0', '1':'1', '6':'9', '8':'8', '9':'6'}
        i, j = 0, len(num)-1
        while i<=j:
            if not num[i] in map or map[num[i]] != num[j]:
                return False
            i, j = i+1, j-1
        '''
        return True

    # lintcode Easy 1632 · Strobogrammatic Number 考 map + 字符串 的处理，字符串处理搞得我有点昏
    def countGroups(self, emails):
        dic = {}
        counter = 0

        for each in emails:
            temp = self.process_each_email(each)

            if temp in dic:
                dic[temp] += 1
                if dic[temp] == 2: # 在这里计数，可以降低时间复杂度
                    counter += 1
            else:
                dic[temp] = 1

        return counter
    def process_each_email(self, email):
        before_at = True
        index = 0
        res = []

        while index < len(email):
            if email[index] == '@':
                before_at = False
                res.append(email[index])
                index += 1
                continue

            if before_at:
                if email[index] == '.':
                    index += 1
                    continue
                if email[index] == '+':
                    index += 1
                    while index < len(email) and email[index] != '@':
                        index += 1
                    if email[index] == '@':
                        before_at = False
                        res.append(email[index])
                        index += 1
                        continue
                res.append(email[index])
                index += 1
            else:
                res.append(email[index])
                index += 1

        return ''.join(res)

    # lintcode(力扣299) Medium 1299 · Bulls and Cows
    def getHint(self, secret, guess):
        """时间O(n) 空间O(1)"""
        if not secret or not guess:
            return '0A0B'

        dic = collections.Counter(secret)

        bulls = 0  # 位置对，值对
        cows = 0  # 值对，位置不对

        for i, c in enumerate(guess):
            if c in dic:
                dic[c] -= 1

                if dic[c] <= 0:
                    dic.pop(c)

                if i < len(secret) and guess[i] == secret[i]:
                    bulls += 1
                else:
                    cows += 1

        return "{}A{}B".format(bulls, cows)

    # lintcode(力扣128) Medium 124 · Longest Consecutive Sequence
    def longestConsecutive_Time_Limit_Exceeded(self, num):
        """我自己的答案，超时了"""
        dic = {}

        for n in num:
            dic[n] = n + 1

        max_len = -float('inf')

        for n in num:
            count = 1
            val = dic.get(n, None)

            while val is not None and val in dic:
                count += 1
                val = dic.get(val, None)

            max_len = max(count, max_len)

        return max_len

    # lintcode(力扣128) Medium 124 · Longest Consecutive Sequence
    def longestConsecutive(self, nums) -> int:
        """
        真正的O(N)O(N)解

        而且不需要动态的从哈希或set里删除元素。

        这是一个计数题，别的解法都把它弄复杂了。假设你已经掌握了dict和set，往下看：

        拿那个[100, 4, 200, 1, 3, 2]样例，你该怎么数数呢？你先从100数，然后呢，就没有了。再从4开始数，唉，不对，不应该，因为后面还有3，2，1
        所以应该把4跳过去，待会从小的数开始数。再后面是200，因为没有199，所以应该从200开始。

        或者这样看，每一个连续序列都可以被这个序列的最小值代表，要找到最小值才开始数，这样无重复，才能做到O(N)O(N)。

        具体来看，这个代码做了三个NN的操作：

        建dict
        for循环里，看每一个数字n是否有n-1存在
        while循环，从小到大的数连续序列

        如果用先排序一下的话就始终 nlogn了
        """
        max_len, table = 0, {num: True for num in nums}

        for lo in nums:
            if lo - 1 not in table:
                # 能走到这里说明 lo-1 已经是比最小的还小1
                curr = lo + 1
                while curr in table:
                    curr += 1
                                        # 长度
                max_len = max(max_len, curr - lo)

        return max_len


if __name__ == '__main__':

    sol = Solution()
    l = sol.longestConsecutive2([1,2,0,1])
    print(l)
    pass