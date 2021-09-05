
class Solution:

    # 力扣 Medium 1041. Robot Bounded In Circle 元组的使用
    def isRobotBounded(self, instructions: str) -> bool:
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dirIndex = 0
        position, transitionDir = (0, 0), dirs[dirIndex]
        for char in instructions:
            if char == 'G':
                # 元组相加
                position = tuple(map(lambda i, j: i + j, position, transitionDir))
            elif char == 'L':
                dirIndex -= 1
                transitionDir = dirs[dirIndex % 4]
            else:
                dirIndex += 1
                transitionDir = dirs[dirIndex % 4]
        return transitionDir != (0, 1) or position == (0, 0)

    # 力扣 E 937. Reorder Data in Log Files 难点：字符串操作 + 多权重排序  自己写的乱七八糟版
    def reorderLogFiles1(self, logs):
        for i in range(len(logs)):
            temp = logs[i]
            logs[i] = (logs[i].split(), temp)

        logs.sort(key=lambda e: e[0][1].isdigit())
        index = 0
        for i, char in enumerate(logs):
            if char[0][1].isdigit():
                index = i
                break
        log2 = logs[0:i]
        log2.sort(key=lambda x: (x[0][1:], x[0][0]))
        for i in range(0, index):
            logs[i] = log2[i]

        for i in range(len(logs)):
            logs[i] = logs[i][1]

        return logs

    # 力扣 E 937. Reorder Data in Log Files 难点：字符串操作 + 自己定义多权重排序   代码规整版
    def reorderLogFiles2(self, logs):
        """
        给sorted(key)里的key定义一个规则：key1，key2，key3  先按key1排序，再按key2排序，再按key3排序

        N be the number of logs in the list and M be the maximum length of a single log.
        时间复杂度是O(MNlogN)
        """
        def get_key(log):
            _id, content = log.split(maxsplit=1)   #  注意活用 str.split(sep=None, maxsplit=-1) 函数
            return (0, content, _id) if content[0].isalpha() else (1, None, None)
            # key_1: this key serves as a indicator for the type of logs.
            #        For the letter-logs, we could assign its key_1 with 0,
            #        and for the digit-logs, we assign its key_1 with 1.
            # key_2: for this key, we use the content of the letter-logs as its value,
            #        so that among the letter-logs, they would be further ordered based on their content
            # key_3: similarly with the key_2, this key serves to further order the letter-logs.
            #        We will use the identifier of the letter-logs as its value, so that for the
            #        letter-logs with the same content, we could further sort the logs based on its identifier,
            # 注意 digit-logs不需要key_2和key_3, 因此如果识别出是 digit-logs，返回的是 (1, None, None)

        return sorted(logs, key=get_key)  #  这句好像不能写成 in-place 的这样 logs.sort(logs, key=get_key)

    # 力扣 M 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts  自己写的代码有点乱
    def maxArea1(self, h: int, w: int, horizontalCuts, verticalCuts):
        verticalCuts.sort()

        w1 = -float('inf')
        for i in range(len(horizontalCuts) + 1):
            if i == 0:
                temp = horizontalCuts[i] - 0
            elif i == len(horizontalCuts):
                temp = h - horizontalCuts[i - 1]
            else:
                temp = horizontalCuts[i] - horizontalCuts[i - 1]

            w1 = max(w1, temp)

        w2 = -float('inf')
        for i in range(len(verticalCuts) + 1):
            if i == 0:
                temp = verticalCuts[i] - 0
            elif i == len(verticalCuts):
                temp = w - verticalCuts[i - 1]
            else:
                temp = verticalCuts[i] - verticalCuts[i - 1]

            w2 = max(w2, temp)

        # Don't forget the modulo 10^9 + 7   be careful of overflow
        # Python doesn't need to worry about overflow.  Don't forget the modulo though!
        return w1 * w2 % (10 ** 9 + 7)

    # 力扣 M 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts  代码更clean
    def maxArea2(self, h: int, w: int, horizontalCuts, verticalCuts):
        """
        时间复杂度 O(N⋅log(N)+M⋅log(M))
        空间复杂度 O(1)
        """
        horizontalCuts.sort()
        verticalCuts.sort()

        # Consider the edges first
        max_height = max(horizontalCuts[0], h - horizontalCuts[-1])
        for i in range(1, len(horizontalCuts)):
            # horizontalCuts[i] - horizontalCuts[i - 1] represents the distance between
            # two adjacent edges, and thus a possible height
            max_height = max(max_height, horizontalCuts[i] - horizontalCuts[i - 1])

        # Consider the edges first
        max_width = max(verticalCuts[0], w - verticalCuts[-1])
        for i in range(1, len(verticalCuts)):
            # verticalCuts[i] - verticalCuts[i - 1] represents the distance between
            # two adjacent edges, and thus a possible width
            max_width = max(max_width, verticalCuts[i] - verticalCuts[i - 1])

        # Python doesn't need to worry about overflow - don't forget the modulo though!
        return (max_height * max_width) % (10 ** 9 + 7)

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

        patternCount = collections.defaultdict(int)   # 它的value是integer
        for user in userHistory.keys():
            temp = combinations(userHistory[user], 3)   # 注意生成的组合竟然是 stable 的！！！
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

        def sortKey(pattern):   # pattern是字典里的每一个 key元素
            return (-patternCount[pattern], pattern)

        # 为啥字典可以排序，因为Dictionaries are ordered in Python 3.6
        # 返回的 sorted_patternCount 是按 key=sortKey 规则后 的 keys的list
        sorted_patternCount = sorted(patternCount, key=sortKey)
        return sorted_patternCount[0]

    def mostVisitedPattern2(self, username, website, timestamp):
        import collections
        from itertools import combinations

        users = collections.defaultdict(list)

        for user, time, site in sorted(zip(username, timestamp, website), key=lambda x: (x[0], x[1])):
            users[user].append(site)

        patterns = collections.Counter()

        for user, sites in users.items():
            patterns.update(collections.Counter(set(combinations(sites, 3))))

        # 先按照patterns的key排个序生成 keys的list
        temp = sorted(patterns)  #  或等于 temp = sorted(patterns.keys())
        # 再按 patterns 里的 values 里找最大value 对应的 temp 里的 key
        # 如果value相同，就找第一个出现的 （而谁先出现，上句代码已经sort过了）
        maxi = max(temp, key= lambda x : patterns[x])  # 也可以写成：= max(temp, key=patterns.get)
        return maxi

    # 力扣 M 1167. Minimum Cost to Connect Sticks 不难用minHeap就好, 主要是知道怎么用 heapq
    def connectSticks(self, sticks):
        import heapq
        res = 0

        # in-place, O(N) in linear time  所以好像这个写法不耗空间
        heapq.heapify(sticks)

        while len(sticks) > 1:
            # pop 和 push 都是 O(logN)
            # 但大概要 adding/popping (N-1) elements to the priotity queue 所以总的时间复杂度位O(NlogN)
            temp = heapq.heappop(sticks) + heapq.heappop(sticks)
            res += temp
            heapq.heappush(sticks, temp)

        return res

    # 力扣 M 1792. Maximum Average Pass Ratio  通过 负号 把 minHeap结构转成 maxHeap
    def maxAverageRatio(self, classes, extra):
        import heapq
        #  通过 负号 把 minHeap结构转成 maxHeap
        ratios = [(-(c[0] + 1) / (c[1] + 1) + c[0] / c[1], c) for c in classes]
        heapq.heapify(ratios)

        for _ in range(extra):
            c = heapq.heappop(ratios)
            passes = c[1][0] + 1
            total = c[1][1] + 1

            heapq.heappush(ratios, (-((passes + 1) / (total + 1) - passes / total), [passes, total]))

        s = 0
        for c in ratios:
            s += c[1][0] / c[1][1]

        return s / len(ratios)

    # 力扣 E 1710. Maximum Units on a Truck
    def maximumUnits(self, boxTypes, truckSize):
        boxTypes.sort(key=lambda x: x[1], reverse=True)
        res = 0
        length = len(boxTypes)
        iterator = iter(boxTypes)
        i = 0
        # 写法1：把list变成iter一个一个看
        while truckSize > 0 and i < length:
            box = next(iterator)
            if truckSize >= box[0]:
                res += box[0] * box[1]
                truckSize -= box[0]
            else:
                res += truckSize * box[1]
                truckSize = 0
            i += 1

        '''
        # 写法2
        for box in boxTypes:
            if truckSize >= box[0]:
                res += box[0] * box[1]
                truckSize -= box[0]
            else: 
                res += truckSize * box[1]
                truckSize = 0
                break
        '''
        return res

    # 力扣 M 1268. Search Suggestions System 这种是比较暴力解法，时间复杂度较高。有一种 trie + dfs解法,等学到再做一便
    def suggestedProducts(self, products, search):
        products.sort()
        res = []
        for i in range(len(search)):
            search_prefix = search[0:i+1]
            temp = []
            for product in products:
                product_prefix = product[0:i+1]
                if search_prefix == product_prefix:
                    temp.append(product)
                    if len(temp) == 3:
                        break
            res.append(temp)
        return res


if __name__ == '__main__':



    words = ["cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"]
    sol = Solution()
    ans = sol.findAllConcatenatedWordsInADict(words)
    print(ans)