class Solution:
    """
    Dijkstra 专门用来处理 weighted图的 shortest distance
    再配合 minHeap 或 maxHeap 来做特别好用

    简述dfs,bfs,Dijkstra思想及区别
    https://blog.csdn.net/qq_16949707/article/details/51490764

    [heap]库必记的知识点
    1.导包写 from heapq import heapify, heappop, heappush
    2. heapify时间复杂度是O(N), heappop和heappush时间复杂度是O(logN)
    3. 按什么元素来计算min或max？按一个元素的列来计算，比如 (x，y，z) 先比x再比y再比z
    4. 一个普通list要先heapify才能正常pop和push；但如果是空list，就直接pop和push了
    5. 这俩maybe好用
        List2 = heapq.nlargest(n ,List1, key=lambda x: x[1]) 从 List1 中返回前n个最大的元素组成一个新的List返回，以什么为标准排序key定
        List1 不用先 heapify，当然如果只返回1个数字的话还是用 min() max() 效率高
        List2 = heapq.nsmalles(n ,List1, key=lambda x: x[1]) 从 List1 中返回前n个最小的元素组成一个新的List返回，以什么为标准排序key定
    """

    # 力扣M 505 The Maze II 复杂图的最短路径，用Dijkstra来解决的（更简单的看写法2）
    def shortestDistance1(self, maze, start, destination) -> int:
        """
        1.就着example把跟着代码走一遍，好难啊
        2.除了层级，还要顾到走了多少步. 所以这不是一个简单图！！！不能用层级来算步数！！
        3.edges的weight不一样，算是复杂图，用 Dijkstra 解法
            把所有 nodes 的 step 设置为 无限大
            除非遇到 node 的 step 更小了，再更新这个 node 的 step
            （Dijkstra解决不了edge weight是负数的那种）

        """
        from collections import deque

        # 把 start 和 destination 变成 tuple
        start = tuple(start)
        destination = tuple(destination)

        # initialize all possible spaces' steps as infinite
        steps = self.initialize_posible_spaces_infinite(maze)

        # 开始 BFS
        queue = deque([start])
        steps[start] = 0  # 起点的step设为0

        while queue:
            current_node = queue.popleft()

            # 返回的 next_nodes 是个字典的数据结构
            next_nodes_new_step = self.to_next_nodes_steps(maze, current_node, steps[current_node])

            for next_node in next_nodes_new_step:
                # 如果 原来的 next_node 的 step 已经小于等于 new step了，就跳过，处理这个点
                if steps[next_node] <= next_nodes_new_step[next_node]:
                    continue

                # 发现 new node 的 new step 比原来 step 更小，再处理
                steps[next_node] = next_nodes_new_step[next_node]
                queue.append(next_node)

        if steps[destination] == float('inf'):
            return -1
        else:
            return steps[destination]
    def initialize_posible_spaces_infinite(self, maze):
        steps = {}
        for x in range(len(maze)):
            for y in range(len(maze[0])):
                if maze[x][y] == 0:
                    steps[(x, y)] = float('inf')
        return steps
    def to_next_nodes_steps(self, maze, this_node, pre_steps):
        next_nodes = {}
        x, y = this_node
        DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in DIRECTIONS:
            next_x = x
            next_y = y
            count_steps = 0

            while self.is_valid788(next_x + dx, next_y + dy, maze):  # 这里加一个在不在dict里的检查
                next_x += dx
                next_y += dy
                count_steps += 1

            if count_steps > 0:
                next_nodes[(next_x, next_y)] = pre_steps + count_steps

        return next_nodes
    def is_valid788(self, x, y, maze):
        # 没有撞墙也没有走出地图就是 valid
        if not (0 <= x < len(maze) and 0 <= y < len(maze[0])):
            return False

        if maze[x][y] == 1:
            return False

        return True

    # 力扣M 505 The Maze II 也是Dijkstra算法，但用了 heap(priority queue)的技巧处理，写得比方法1简洁多了！
    def shortestDistance2(self, maze, start, destination):
        from heapq import heapify, heappop, heappush  # used as priority queue

        q = [(0, start)]  # element: [distance_to_start_point, position] [(0，4)，0]
        done = set()

        while q:
            # 一定要写成这种形式来承接元素
            distance, [x, y] = heappop(q)  #  每次 pop一个当前的最短路径
            if [x, y] == destination:  # 易错点:由于destination是list，所以要[x,y]跟它比，不能(x,y)跟它比，用tuple和list比是永远不会相等的！
                return distance

            if (x, y) in done:  # 小技巧！已经处理过的，就可以直接skip掉，加快速度。虽然不加也不会出错
                continue

            done.add((x, y))  # 注意这是跟一般BFS不一样的地方，一个点处理完了才加到hashmap，不是一visited就加

            possible_next_positions = self.all_possible_next_positions(maze, x, y)

            for steps, position in possible_next_positions:
                if position in done:
                    continue
                heappush(q, (distance + steps, position))

        return -1
    def all_possible_next_positions(self, maze, x, y):
        Directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        possible_next_steps = []

        for dx, dy in Directions:
            next_x = x
            next_y = y
            steps = 0  # 易错点:这个steps不能漏掉，这是每个edge的重量
            while self.is_valid(next_x + dx, next_y + dy, maze):
                next_x += dx
                next_y += dy
                steps += 1

            if steps != 0:  # 易错点：这一步的检查很关键，如果有走动就加进List，由于已经在前四句检查过是否valid，这里不需要检查了
                possible_next_steps.append((steps, (next_x, next_y)))

        return possible_next_steps
    def is_valid(self, x, y, maze):
        # ensur it's within range
        if x < 0 or x >= len(maze):
            return False
        if y < 0 or y >= len(maze[0]):
            return False

        # ensure it's not a wall
        return maze[x][y] != 1

    # 亚麻intern VO题, 力扣M 1514. Path with Maximum Probability一开始很抽象，转化成shortest distance概念用 Dijkstra + heap技巧来解的 跟力扣505解法2类似
    def maxProbability(self, num: int, edges, probabilities, start: int, end: int) -> float:
        """
        这道题是求最大概率，如何转化成shortest distance？
        由于概率<1, 乘得越多，积越小
        我们要走最少的步数，每步要走得尽量大
        因此在每一轮中，要先找最大概率那个处理，这道题就可以转化为求重量图的最短路径的题，用 dijkstra
        """
        from heapq import heapify, heappop, heappush  # used as priority queue

        graph, weights = self.build_graph_weights(num, edges, probabilities)

        q = [(-1, start)]  # 以第一个元素大小为priority，为什么是-1，通过负号变成 max_heap了
        done = set()
        while q:
            product, n = heappop(q)

            if n == end:
                return -product # 注意，返回的时候要negative一下

            if n in done:       # 小技巧！已经处理过的，就可以直接skip掉，加快速度。虽然不加也不会出错
                continue

            done.add(n)   # 注意这是跟一般BFS不一样的地方，一个点处理完了才加到hashmap，不是一visited就加

            for next_n in graph.get(n, []):
                if next_n in done: continue
                next_product = product * weights.get((n, next_n), 0)
                heappush(q, (next_product, next_n))
        return 0
    def build_graph_weights(self, n, edges, succProb):
        """
        graph:
        0 {1,2}
        1 {0,2}
        2 {1,0}

        weights:
        (0,1)  0.5
        (1,0)  0.5
        (1,2)  0.5
        (2,1)  0.5
        (0,2)  0.2
        (2,0)  0.2

        """
        graph = {}
        weights = {}
        for i in range(n):
            graph[i] = set()

        for i, (node1, node2) in enumerate(edges):
            #   元素这样写是可以的
            graph[node1].add(node2)
            graph[node2].add(node1)
            weights[(node1, node2)] = succProb[i]
            weights[(node2, node1)] = succProb[i]

        return graph, weights

if __name__ == '__main__':
    l = [(1,2,3), (3,1.5,1), (4,5,6)]
    import heapq
    from heapq import heapify, heappop, heappush
    # heapify(l)
    a= heapq.nlargest(2,List, key=lambda x: x[1])
    sol = Solution()
    res = sol.maxProbability(3, [[0,1],[1,2],[0,2]], [0.5,0.5,0.2], 0, 2)
    print(res)