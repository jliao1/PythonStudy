# 亚麻intern VO题 力扣M 981 Time Based Key-Value Store hashmap做法蛮简单的，算easy吧
class TimeMap1:

    def __init__(self):
        self.Map = {}
        self.min_timestamp = float('inf')

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.Map:
            self.Map[key] = {}
            self.Map[key][timestamp] = value
        else:  # key in Map
            self.Map[key][timestamp] = value

        self.min_timestamp = min(self.min_timestamp, timestamp)

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.Map:
            return ''
        while timestamp >= self.min_timestamp:
            value = self.Map[key].get(timestamp, None)
            if value is None:
                timestamp -= 1
            else:
                return value
        return ''

# 亚麻intern VO题 力扣M 981 Time Based Key-Value Store 二分法提速，但我二分法写得不对，不知怎么回事
class TimeMap2(object):

    def __init__(self):
        import collections
        self.dic = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        self.dic[key].append([timestamp, value])

    # 二分法这样写不对……不知为啥
    def get(self, key, timestamp):
        if key not in self.dic:
            return ''

        arr = self.dic[key]
        n = len(arr)

        left = 0
        right = n - 1

        while left + 1 < right:
            mid = (left + right) // 2
            e = arr[mid]
            if e[0] == timestamp:
                return e[1]
            elif e[0] < timestamp:
                right = mid
            else:  # e[0] > timestamp:
                left = mid

        if arr[left][0] == timestamp:
            return arr[left][1]
        if arr[right][0] == timestamp:
            return arr[right][1]

        return arr[left][1]

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    #亚麻intern VO题 力扣M 253 Meeting Rooms II Priority queue的解法，双指针解法没看懂，学一下
    def minMeetingRooms1(self, intervals):
        intervals.sort(key=lambda x: x[0])
        from heapq import heappush, heappop, heapify
        q = [intervals[0][1]]  # 30

        for each in intervals[1:]:
            earlist_ending_time = q[0]  # 30

            if each[0] < earlist_ending_time:
                heappush(q, each[1])
            else:
                heappop(q)
                heappush(q, each[1])
        return len(q)

    # 亚麻intern VO题 力扣E 852 Peak Index in a Mountain Array  这是O(n)时间复杂度
    def peakIndexInMountainArray(self, arr) :
        for i in range(1,len(arr)-1):  #注意这个一定不要写成 range(1: len(arr)-1)是逗号！
            if arr[i - 1] < arr[i] and arr[i] >= arr[i + 1]:
                return i

        return -1

    # 亚麻intern VO题 力扣E 852 Peak Index in a Mountain Array  这是O(logN)时间复杂度，但我好像写错了
    def peakIndexInMountainArray(self, arr) :
        l = 0
        r = len(arr) - 1

        while l + 1 < r:
            mid = (l + r) // 2
            if arr[mid - 1] > arr[mid] > arr[mid + 1]:
                r = mid
            elif arr[mid - 1] < arr[mid] < arr[mid + 1]:
                l = mid
            elif arr[mid - 1] < arr[mid] and arr[mid] > arr[mid + 1]:
                return mid

    # 亚麻intern VO题 M 1448. Count Good Nodes in Binary Tree 这题对我算Easy，但写得还不算简练
    def goodNodes(self, root):
        """时间空间 O(N) """
        if not root:
            return 0

        self.count = 1

        if root.left:
            self.helper1448(root.left, root.val)
        if root.right:
            self.helper1448(root.right, root.val)

        return self.count
    def helper1448(self, root, current_max):
        X = root.val
        if X >= current_max:
            self.count += 1

        current_max = max(X, current_max)  # 易错点，这一步忘记了

        if root.left:
            self.helper1448(root.left, current_max)
        if root.right:
            self.helper1448(root.right, current_max)

    # 亚麻intern VO， Jason 给我 mock 题
    def getRelevantArcs(self, Warehouses, Arcs):
        graph = self.build(Arcs)
        from collections import deque
        results = []

        for Warehouse in Warehouses:
            q = deque([Warehouse])
            vistied = set([Warehouse])

            while q:
                place = q.popleft()
                next_warehouses = graph.get(place,[])  # (C, B)

                for next_warehouse in next_warehouses:
                    if next_warehouse in vistied:
                        continue
                    q.append(next_warehouse)
                    vistied.add(next_warehouse)
                    results.append((next_warehouse, place))

        return results
    def build(self, Arcs):
        d = {}  # route graph

        for each in Arcs:
            start = each[0]
            end = each[1]
            if end not in d:
                d[end] = set()
                d[end].add(start)
            else:
                d[end].add(start)


        return d

    # 亚麻intern VO题，力扣M 17 Letter Combinations of a Phone Number 我用双端队列做的，back tracking不懂怎么做
    def letterCombinations(self, digits: str):
        if not digits:
            return []
        to_letter = {
            '2': set(['a', 'b', 'c']),
            '3': set(['d', 'e', 'f']),
            '4': set(['g', 'h', 'i']),
            '5': set(['j', 'k', 'l']),
            '6': set(['m', 'n', 'o']),
            '7': set(['p', 'q', 'r', 's']),
            '8': set(['t', 'u', 'v']),
            '9': set(['w', 'x', 'y', 'z'])
        }
        from collections import deque
        # 用双端队列做
        result = deque([''])
        for number in digits:

            current_len = len(result)

            for _ in range(current_len):
                #               3
                current_str = result.popleft()
                for char in to_letter[number]:
                    new_str = current_str + char  # 易错点：这里不能写成
                    result.append(new_str)

        return list(result)

    # 力扣MH 36 Valid Sudoku 提炼函数思维 这个sub-Box是3X3,如果是KXK也一样做
    def isValidSudoku(self, board): # borad是个二维数组
        """
        时间复杂度 O(n^2)  空间复杂度O(n)因为只存

        Input: board = #如果第一个8是5，那就return true了
                    [["8","3",".",".","7",".",".",".","."]
                    ,["6",".",".","1","9","5",".",".","."]
                    ,[".","9","8",".",".",".",".","6","."]
                    ,["8",".",".",".","6",".",".",".","3"]
                    ,["4",".",".","8",".","3",".",".","1"]
                    ,["7",".",".",".","2",".",".",".","6"]
                    ,[".","6",".",".",".",".","2","8","."]
                    ,[".",".",".","4","1","9",".",".","5"]
                    ,[".",".",".",".","8",".",".","7","9"]]
                Output: false
        """
        rows = len(board)
        cols = len(board[0])

        # check each column
        for col in range(cols):
            Set = set()
            for row in range(rows):
                element = board[row][col]
                if not self.is_valid36(Set, element):
                    return False

        # check each row
        for row in board:
            Set = set()
            for i in range(len(row)):
                element = row[i]
                if not self.is_valid36(Set, element):
                    return False

        k = 3
        # check sub-box
        for i in range(k):
            for j in range(k):
                # i: 0 1 2   j: 0 1 2
                Set = set()
                for row in range(i * k, i * k + k):
                    # row:  (0,3) (3,6) (6,9)
                    # col:  (0,3) (3,6) (6,9)
                    for col in range(j * k, j * k + k):
                        element = board[row][col]
                        if not self.is_valid36(Set, element):
                            return False

        return True
    # 提炼成检查是否valid的函数. 什么时候valid？当element不是‘.’并且是1-9第一次出现在Set里
    def is_valid36(self, Set, element):
        if element == '.':
            return True
        elif element not in Set:
            Set.add(element)
            return True
        else:
            return False

    # Blend面经题 跟领扣14一样的  这是力扣E 278 First Bad Version
    def firstBadVersion(self, n):
        """
        # The isBadVersion API is already defined for you.
        # @param version, an integer
        # @return an integer
        # def isBadVersion(version):
        """
        start = 1
        end = n

        while start + 1 < end:
            mid = (start + end) // 2
            if not isBadVersion(mid):
                start = mid
            else:
                end = mid

        if isBadVersion(start):
            return start
        if isBadVersion(end):
            return end

        return -1

    # Blend VO题 力扣M54 Spiral Matrix 螺旋遍历matrix  技巧是direction%4来走圈圈
    def spiralOrder(self, matrix):
        if matrix == []:
            return []

        upper_bound = 0
        lower_bound = len(matrix) - 1
        left_bound = 0
        right_bound = len(matrix[0])-1
        direct = 0  # 0: go right   1: go down   2: go left   3: go up
        res = []

        while True:
            # 每处理完一个方向，bound也要改变
            if direct == 0:  # go right
                for i in range(left_bound, right_bound+1):
                    res.append(matrix[upper_bound][i])
                upper_bound += 1
            if direct == 1:  # go down
                for i in range(upper_bound, lower_bound+1):
                    res.append(matrix[i][right_bound])
                right_bound -= 1
            if direct == 2:  # go left
                for i in range(right_bound, left_bound-1, -1):
                    res.append(matrix[lower_bound][i])
                lower_bound -= 1
            if direct == 3:  # go up
                for i in range(lower_bound, upper_bound-1, -1):
                    res.append(matrix[i][left_bound])
                left_bound += 1

            # while 暂停的条件
            if upper_bound > lower_bound or left_bound > right_bound: return res

            # 绕圈写成这样的小技巧
            direct = (direct+1) % 4

    # Blend VO题 力扣M166 Fraction to Recurring Decimal 字符串的骚操作 5星
    def fractionToDecimal(self, numerator, denominator):
        if numerator % denominator == 0:
            return str(numerator // denominator)  # 只要可以被整除，就可以返回

        sign = "-" if numerator * denominator < 0 else ""  # 设置 正负号
        num, den = abs(numerator), abs(denominator)        # 搞成positive number

        quotient, reminder = divmod(num, den)  # 返回的是商和余数  quotient 和 reminder
        res = sign + str(quotient) + "."

        # 纪录 rem 对应的位置
        num_to_pos = {}

        # 如果 reminder不为0，或者 reminder重复了就直接跳出
        while reminder and reminder not in num_to_pos:
            num_to_pos[reminder] = len(res)
            quotient, reminder = divmod(10 * reminder, den)
            res += str(quotient)

        if reminder in num_to_pos:
            # 如果rem重复，在对应的位置 插入 "("
            index = num_to_pos[reminder]
            res = res[:index] + "(" + res[index:] + ")"

        return res

    # Blend VO题 不知道考的是哪题
    def move(self, grid):
        def robotMove(grid, i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                return 0
            if grid[i][j] == 1:
                return 0
            if i == len(grid) - 1 and j == len(grid[0]) - 1:
                return 1
            import copy
            arr = copy.deepcopy(grid)
            arr[i][j] = 1
            return robotMove(arr, i - 1, j) + robotMove(arr, i, j + 1) + robotMove(arr, i, j - 1) + robotMove(arr, i + 1, j)
        return robotMove(grid, 0,0)








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

    l1 = [node3,node5,node8,node7]
    l1.sort()

    sol = Solution()
    res = sol.mergeKLists([node1,node4,node7])

    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    res = sol.move(grid)
    print(res)

    # ans = sol.maxArea(input2)
    # print(ans)