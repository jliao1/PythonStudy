"""
[对于字符的排序]
words1 = ["apple", "app"] #                  短的在前
words2 = sorted(words1)   # words2 = 排序后是 ["app", "apple"]

"""

# build-in type不能重载  运算符重载 (Operator overloading)

# 对类 object 的比较符号重载，写法1: (class内)  heapq,sort()，min(),max()都可以直接用 不用写(key = lambda x : x[0])
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    # # 对一个 object 的比较符号重载，写法1： 可以写在class里
    # def __lt__(list_node_1, list_node_2):
    #     return list_node_1.val < list_node_2.val

# 对类 object 的比较符号重载，写法2： 这写在了 ListNode 的 class外 （用法和写法1一样） # used by 力扣23
ListNode.__lt__ = lambda node1, node2: (node1.val < node2.val) # 更小的要排在前(因为排序是从 小-> 大的)

# 重载int排序规则，写法1 (class内)    这种heapq不能直接用,因为sort() min() max()在用的时候要写(key = lambda x : x[0])  # used by 力扣179
class LargerNumKey(int):
    def __lt__(x, y):    # 比如 x=20, y=2
        #                       202 < 220  这种情况是 x小y大，所以x排在y前
        return  int( str(x)+str(y) ) < int( str(y)+str(x) )   # 更小的要排在前(因为排序是从 小-> 大的)

# 重载int排序规则，写法2 (class外)
# LargerNumKey.__lt__=lambda  x,y: int( str(x)+str(y) ) < int( str(y)+str(x) )

class Solution:

    # 定义一类object的比较规则定义,heapq直接用 - H力扣23 Merge k Sorted Lists 用 heapq 做的    Blend VO题
    def mergeKLists(self, lists):
        if not lists:
            return None

        from heapq import heappush, heappop
        dummy = ListNode(0)
        tail = dummy
        heap = []
        for head in lists:
            if head:
                heappush(heap, head)

        while heap:
            head = heappop(heap)
            tail.next = head
            tail = head
            if head.next:
                heappush(heap, head.next)

        return dummy.next

    # 重载int组合的比较规则,用时必须(key = LargerNumKey) - M力扣179(领扣184) · Largest Number
    def largestNumber(self, nums):
        """
        来分析下复杂度
        【时间】 Although we are doing extra work in our comparator,
        it is only by a constant factor.
        Therefore, the overall runtime is dominated by the
        complexity of sort, which is O(NlogN) in Python and Java.
        【空间】 we allocate O(n) additional space to store 排序后的列表和连结后的string
        """
        nums.sort(key = LargerNumKey, reverse = True)  # 重载 int 的比较规则
        res = ''
        for c in nums:
            res += str(c)

        return '0' if res[0] == '0' else res

    # 力扣 M 1152. Analyze User Website Visit Pattern  又是对list+字典+排序等一系列骚操作  简介一点的写法看方法2
    def mostVisitedPattern1(self, username, website, timestamp):
        import collections
        from itertools import combinations

        TUW = tuple(zip(timestamp, username, website))
        sortedTUW = sorted(TUW)
        # (1, u'joe', u'home')

        userHistory = collections.defaultdict(list)  # 它的values是list
        for time, user, website in sortedTUW:
            userHistory[user].append(website)
        '''
        以上3行可以这样写（1）： 不用导包
        userHistory={}
        for time,user,website in sortedTUW:
            if user in userHistory: 
                userHistory[user].append(website)
            else:
                userHistory[user] = [website]

        以上3行还可以这样写（2）：不用导包
        userHistory={}
        for time,user,website in sortedTUW:
            userHistory.setdefault(user,[])
            userHistory[user].append(website)

        最终出来的效果是：
        {
        u'james': [u'home', u'cart', u'maps', u'home'],
        u'joe': [u'home', u'about', u'career'],
        u'mary': [u'home', u'about', u'career']   }
        '''

        patternCount = collections.defaultdict(int)  # 它的value是integer
        for user in userHistory.keys():
            temp = combinations(userHistory[user], 3)  # 注意生成的组合竟然是 stable 的！！！
            combs = set(temp)
            for comb in combs:
                patternCount[comb] = patternCount[comb] + 1
        '''
        以上6句子还可以写成（1）
        patternCount = {}
        for user in userHistory.keys():
            temp = combinations(userHistory[user], 3)
            combs = set(temp)
            for comb in combs:    
                if comb not in patternCount:
                    patternCount[comb] = 1
                else:
                    patternCount[comb]=patternCount[comb]+1

        以上6句子还可以写成（2）：
        patternCount={}
        for user in userHistory.keys():
            temp = itertools.combinations(userHistory[user], 3)
            combs = set(temp)
            for comb in combs:    
                patternCount.setdefault(comb, 1)
                patternCount[comb]=patternCount[comb]+1
        '''

        def sortKey(pattern):  # pattern是字典里的每一个 key元素
            return (-patternCount[pattern], pattern)

        # 为啥字典可以排序，因为Dictionaries are ordered in Python 3.6
        # 返回的 sorted_patternCount 是按 key=sortKey 规则后 的 keys的list
        sorted_patternCount = sorted(patternCount, key=sortKey)
        return sorted_patternCount[0]

    def mostVisitedPattern2(self, username, website, timestamp):
        import collections
        from itertools import combinations

        users = collections.defaultdict(list)
                                                                                        # 先按第一列排序，再按第二列排序
        for user, time, site in sorted(zip(username, timestamp, website), key=lambda x: (x[0], x[1])):
            users[user].append(site)

        patterns = collections.Counter()

        for user, sites in users.items():
            patterns.update(collections.Counter(set(combinations(sites, 3))))

        # 先按照patterns的key排个序生成 keys的list
        temp = sorted(patterns)  # 或等于 temp = sorted(patterns.keys())
        # 再按 patterns 里的 values 里找最大value 对应的 temp 里的 key
        # 如果value相同，就找第一个出现的 （而谁先出现，上句代码已经sort过了）
        maxi = max(temp, key=lambda x: patterns[x])  # 也可以写成：= max(temp, key=patterns.get)
        return maxi


if __name__ == '__main__':

    node1 = ListNode(1)
    node2 = ListNode(4)
    node3 = ListNode(5)
    node1.next = node2
    node2.next = node3

    node4 = ListNode(1)
    node5 = ListNode(3)
    node6 = ListNode(4)
    node4.next = node5
    node5.next = node6

    node7 = ListNode(2)
    node8 = ListNode(6)
    node7.next = node8

    ListNode_List = [node3,node5,node8,node7]
    a = min(ListNode_List)
    ListNode_List.sort()

    sol = Solution()
    res1 = sol.mergeKLists([node1,node4,node7])
    print(res1)

    res2 = sol.largestNumber([1, 20, 2, 23, 4, 8])
    print(res2)

    res3 = sol.insert([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8])
    print(res3)