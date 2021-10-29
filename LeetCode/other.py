
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


    # 力扣 M 1167. Minimum Cost to Connect Sticks 不难用minHeap就好, 主要是知道怎么用 heapq
    def connectSticks(self, sticks):
        # 时间复杂度 O(NlogN)
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

    # 力扣 M 973. K Closest Points to Origin  排序(对lambda的处理)
    def kClosest(self, points, k):

        points.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
        return points[:k]
        # 还可以更简化 return heapq.nsmallest(K, points, lambda (x, y): x * x + y * y) # Sort using heap of size K, O(NlogK)

    # 力扣 M 547. Number of Provinces   连通块问题 matrix + bfs
    def findCircleNum(self, m):
        if not m:  # 输入的list里每个元素可能为空吗？ 英语怎么说
            return 0

        cities = len(m)
        from collections import deque
        num = 0  # number of connected component we find

        queue = deque()
        visited = {}  # key:city  value: num

        for i in range(cities):
            if m[i][i] == 0 or i in visited:
                continue

            start = i
            queue.append(start)
            num += 1
            visited[start] = num

            while queue:
                city = queue.popleft()
                neighbors = self.get_neighbors547(city, m)

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    visited[neighbor] = visited[city]
                    queue.append(neighbor)

        return num
    def get_neighbors547(self, city, m):
        neighbors = []
        for i in range(len(m)):
            if city != i and m[i][city] == 1 and m[city][i] == 1:
                neighbors.append(i)

        return neighbors

    # 力扣 M 323. Number of Connected Components in an Undirected Graph   连通块问题 list + bfs
    def countComponents(self, n: int, edges):
        from collections import deque
        if n < 1: return 0

        # 建图：建所有的node，并链接edges
        graph = self.build323(n, edges)

        visited = set()
        count = 0

        for i in range(n):
        # i 是每个 node，每个node都要访问到
            if i in visited:
                continue
            # 如果 i 不在 visited里，就开始从这个点开始 bfs
            queue = deque([i])
            count += 1   # 能够开始bfs，说明发现了新的一块

            while queue:
                node = queue.popleft()
                neighbors = graph[node]

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    queue.append(neighbor)

        return count
    def build323(self, n, edges):
        graph = {}
        for i in range(n):
            graph.setdefault(i, set())

        for i, j in edges:
            graph[i].add(j)
            graph[j].add(i)

        return graph

    # 力扣 M 200. Number of Islands  自己写的，好难写对啊，速度也很慢，还是看领扣433题答案吧（一样的）
    def numIslands(self, grid):
        visited = set()
        island = 0

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if (x, y) in visited:
                    continue

                if grid[x][y] == '0':
                    visited.add((x, y))
                    continue

                island += 1
                self.bfs200(x, y, visited, grid)

        return island
    def bfs200(self, x, y, visited, grid):
        DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        from collections import deque

        queue = deque([(x, y)])
        visited.add((x, y))

        while queue:
            x, y = queue.popleft()

            for dx, dy in DIRECTIONS:
                next_x = x + dx
                next_y = y + dy
                # visited 是访问过的，而queue里是真的邻居

                # 看是否在 range 里
                if not self.is_withine_range(next_x, next_y, grid):
                    continue

                # 如果在range里，但被访问过了
                if (next_x, next_y) in visited:
                    continue

                # 如果没访问过, 添加到 visited
                visited.add((next_x, next_y))

                # 如果是岛屿
                if grid[next_x][next_y] == '1':
                    queue.append((next_x, next_y))
    def is_withine_range(self, x, y, grid):
        if not 0 <= x < len(grid):
            return False

        if not 0 <= y < len(grid[0]):
            return False

        return True

    # 力扣 H 773. Sliding Puzzle 是典型的BFS但是加了难度……
    def slidingPuzzle(self, board):
        source = self.to_string(board)
        goal = '123450'

        from collections import deque
        seen = {source: 0}
        queue = deque([source])

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return seen[curr]

            neighbors = self.get_neighbors(curr, len(board[0]), len(board))

            for neighbor in neighbors:
                if neighbor in seen:
                    continue

                queue.append(neighbor)
                seen[neighbor] = seen[curr] + 1

        return -1
    def get_neighbors(self, string, width, height):
        neighbors = []

        index_of_0 = string.find('0')

        x = index_of_0 // width # 注意这里要用整除
        y = index_of_0 % width

        Direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in Direction:
            new_x = x + dx
            new_y = y + dy
            # validation:
            if 0 <= new_x < height and 0 <= new_y < width:
                swapping_index = width * new_x + new_y
                temp = list(string)
                temp[index_of_0], temp[swapping_index] = temp[swapping_index], temp[index_of_0]
                neighbors.append(''.join(temp))

        return neighbors
    def to_string(self, s):
        str_list = []
        for List in s:
            #      这里不能忘了把integer变成string
            temp = [str(integer) for integer in List]
            str_list.append(''.join(temp))
        return ''.join(str_list)

if __name__ == '__main__':
    grid = [[4,1,2],[5,0,3]]
    sol = Solution()
    ans = sol.slidingPuzzle2(grid)
    print(ans)