'''
BFS使用场景：
（1）分层遍历，找所有方案（用非递归实现）
    简单图的最短路径 （复杂图还可以用Floyd，dijsktra解决，但这一般都要背可以不学。还有SPFA可以学学）
                    如果问最长路径的话BFS不可以做（一般不会考）
    （分层的意思是，路径有一定方向性，不能绕圈）

（2）连通块的问题(通过一个点找到所有点)    这个一般不用dfs做因为容易stack overflow

（3）扑拓排序(基于有向图), (先修课程问题），是排序，但排的是依赖性

以上3种都可以用dfs做，但最好用bfs做，因为用dfs会stack overflow

为什么BFS可以做最短路径？
图跟树不一样，图可能会有环。但BFS的时候，第二次访问到访问过的点，会获得一条更短路径呢？没有可能。
BFS是把所有走0步就能走到的点放到第一层，把所有走1步就能走到的点放第二层，所有走2步能走到的点放第三层。
所以BFS才可以算最短路径。
用哈希表来记录，每个节点访问情况（如果存在哈希表keys里，就说明访问过了；value记录的是到根的距离）


【拓扑排序】
入度：  in-degree
有向图： Directed Graph （这样才有依赖性）
必须没有环，才可以进行拓扑排序
一个图可能存在多个拓扑序，也可能不存在任何拓扑序
算法描述：
1.统计每个点的入度
2.将入度为0的点(不依赖于任何其他的点)，首先放到队列中去
3.不断从队列里拿出一个点，把这个点连向的点的入度都-1
4.然后把 新的入度为0的点，丢回队列中
重复3和4直到完，那些元素依次出队列(或进队列)的顺序就是拓扑排序
  A ->
        B    左边拓扑排序是 ACB 或 CAB 都是对的其实
  C ->       除非要求你输出的是 字典最小的拓扑排序 就是 ACB 了
            而如果要你按字典序输出的话，普通的queue就要换成priority queue了
问题的问法包括：
（1）求任意一个拓扑排序
（2）是否存在拓扑排序         （这个和上个问题用一份代码就可以解决 领扣616和127）
（3）是否存在且仅存在一个拓扑序 （queue的size要么是0或1，从来不会变成2   代码是领扣605）
（4）按字典序最小，来排拓扑序   （queue用priority queue）
'''
import collections

# 对tree, 三种方法做BFS
class bfsTree:
    # 单队列写BFS lintcode easy 69 · Binary Tree Level Order Traversal
    def levelOrder1(self, root):
        if not root:
            return []

        # step 1 把第一层的节点放到队列里
        queue = collections.deque([root])

        result = []

        # step 2 while 队列非空
        while queue:
            level = []
            # step 3 把上一层的节点放入队列，并拓展出下一层的节点
            size = len(queue)
            for _ in range(size):  # range()函数返回的是一个iterator
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result

    # 单队列写BFS(第二种写法) lintcode easy 69 · Binary Tree Level Order Traversal
    def levelOrder2(self, root):
        if not root:
            return []

        # step 1 把第一层的节点放到队列里
        queue = collections.deque([root])

        result = []

        # step 2 while 队列非空
        while queue:
            # 不一样的是这句
            result.append([node.val for node in queue])

            # step 3 把上一层的节点放入队列，并拓展出下一层的节点
            size = len(queue)
            for _ in range(size):  # range()函数返回的是一个iterator
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result

    # 双队列写BFS(但两个队列其实是用list实现的) lintcode easy 69 · Binary Tree Level Order Traversal
    def levelOrder3(self, root):
        if not root:
            return []

        # step 1 把第一层的节点放到队列里
        queue = [root]
        result = []

        # step 2 while 队列非空
        while queue:
            next_queue = []
            result.append([node.val for node in queue])
            # step 3 把上一层的节点放入队列，并拓展出下一层的节点
            for node in queue:  # range()函数返回的是一个iterator
                if node.left:
                    next_queue.append(node.left)
                if node.right:
                    next_queue.append(node.right)
            queue = next_queue
        return result

    # Dummy node写BFS lintcode easy 69 · Binary Tree Level Order Traversal 这样写的好处是没有for锁进优雅
    def levelOrder(self, root):
        if not root:
            return []

        queue = collections.deque([root, None])
        results, level = [], []

        while queue:
            node = queue.popleft()
            if node is None:
                results.append(level)
                level = []
                if queue:
                    queue.append(None)
                continue

            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return results

# 在图上，做BFS，模版写法，必背！！！！
def bfs(node):              # 注意是用node(地址)来作为keys，而不是node的值，因为node的值是可能重复的，地址才是唯一的
    """
    对graph的BFS，时间复杂度都是O(V+E)  E是边数 V是点数
    一般 E 都比 V 大 所以说是O(E)其实也可以

    为什么呢 因为会每个 node 和它的邻居都会被看1次，
    就相当于是 所有 node被看1次，所有边被看2次 = V + 2E = V + E
    best case: 当是个tree结构的时候，每个node的边是2也就是个常数，所以对于tree做bfs是O(V)
    worst case: 是O(V^2)密集图，两两node都有边相连
    所以要回到本质，要考虑最内层几行代码总共执行多少次（循环每个node和每个node的邻居，相当于把所有的边看了2遍）

    空间复杂度是O(V) 用来存放visited的
    """
    queue = collections.deque([node])  # 要[node]不能只collections.deque(node)，否则报错：'UndirectedGraphNode' object is not iterable
    # distance keys来记录是否访问过，values记录了层级(离起点的距离 )
    distance = {node: 0}  #（如果存在哈希表keys里，就说明访问过了；value记录的是到根的距离）node其实是内存地址
                          # 另外这个就不需要 [node]了
    while queue:
        # 每次拿一个节点出来
        node = queue.popleft()
        # 把这个节点的所有邻居遍历一遍
        for neighbor in node.get_neighbors():
            # 如果这个邻居访问过了，就跳到下个邻居
            if neighbor in distance:
                continue
            # 如果这个邻居没有访问过，就把距离+1 （因为这又是多一层了）
            distance[neighbor] = distance[node] + 1  # 层级关系存储在了 distance 里
            # 然后把这个邻居append进队列
            queue.append(neighbor)

class Solution:  # 令狐冲讲课例题
    # 领扣 M 137·Clone Graph：deep copy一个图。bfs也是解决了找联通块问题
    def cloneGraph(self, node):
        '''
        无向边，通常要存2次，A->B,B->A
        时间 O(V+E)  空间O(V)是主要是visited用的
        '''
        if not node:
            return node
        # step1: find all nods （连通块问题）
        nodes = self.find_all_nodes_by_bfs(node)
        # step2: copy nodes and mapping relationship
        mapping = self.copy_nodes(nodes)
        # step3: copy edges
        self.copy_edges(nodes, mapping)
                # 通过mapping去找到对应的新的节点
        return mapping[node]
    def find_all_nodes_by_bfs(self, node):
        import collections

        # 以下两行代码有点hash queue的意思
        queue = collections.deque([node])  # 要[node]不能只collections.deque(node)，否则报错：'UndirectedGraphNode' object is not iterable
        # node一定要在进queue后就马上加到 visited_set 里；如果在出queue后才加到 visited_set 里，很多元素就重复进队列了，这样会超时
        # 一发现就加到 visited_set里
        visited_set = set([node])  # 要[node]不能只set(node)，否则报错：'UndirectedGraphNode' object is not iterable
                                   # 因为初始化 set和deque时，如果要传入参数，必须是iterable的东西
        while queue:
            # 若要分层的话，这行可以插入加个 for_ in range(len(queue)):  或者用图bfs模版，distance 的 value 就代表了层级
            curr_node = queue.popleft()
            for neighbor in curr_node.neighbors:
                if neighbor in visited_set:
                    continue
                visited_set.add(neighbor)  # 一发现就加到 visited_set里
                queue.append(neighbor)
        return list(visited_set)  # 把set转回list返回
    def copy_nodes(self, nodes):
        mapping = {}
        for node in nodes:
            new_node = UndirectedGraphNode(node.label)
            mapping[node] = new_node
        return mapping
    def copy_edges(self, nodes, mapping):
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_node.neighbors.append(mapping[neighbor])

    # 领扣 H 120·Word Ladder  抽象成最短路径的问题
    def ladderLength(self, start, end, dict_set):
        """
        这题虽然也不是直接graph的题
        但字符串变换，从一个字符串，变到另外一个字符串的，最短变换次数，就是典型的用 简单图的最短路径BFS来解决
        如何抽象？
        将单词看作图中的点，将单词能不能通过一个字母的转换变成另一个单词 看成边
        其中对 string 找邻居 的处理是个 小技巧

        例子
        start ="hit"
        end = "cog"
        dict =["hot","dot","dog","lot","log"]
                    lot - log
        hit - hot <  |     |  > cog
                    dot - dog
        我觉得时间复杂度是O(V*L^2) 但我觉得是O(V*L^3)， 时间复杂度我觉得是O(V+L) 我也不知道我对不对
        """
        dict_set.add(end)  # 因为最后要变成end，所以把end也加dict_set里头好generalize
        from collections import deque

        que = deque([start])
        visited = {start: 1}  # 反正最后也要层数+1，不如直接在这里初始设置为1

        while deque:
            word = que.popleft()

            if word == end:
                return visited[word]

            for neighbor in self.get_word_neighbors(word, dict_set):  # O(V*L^2) 因为对于每个vertex都要获取它的邻居
                if neighbor in visited:                               # O(L)
                    continue
                visited[neighbor] = visited[word] + 1
                que.append(neighbor)

        return 0
    def get_word_neighbors(self, word, dict_set):
        """
        这个部分考对 string 的处理
        这个时间复杂度要很小心，是O(L^2)  L是word的长度
        还有一种代码写法时间复杂度是O(NL) (一般)N是远远大于L的，自然是O(L^2)比O(NL)更快
        空间复杂度是O(L)
        """
        neighbors = []
        for i in range(len(word)):  # 这句是 O(L)
            left = word[:i]  # 这和下句是 O(L)  字符串切片
            right = word[i + 1:]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char != word[i]:
                    maybe_word = left + char + right  # 这句是 O(L), 字符串的组合
                    if maybe_word in dict_set:  # 这句话是 O(L), L 是 maybe_word 的长度
                        neighbors.append(maybe_word)  # 因为哈希表的任何一次增删查改的操作都是O(size of key)，
                        #                               当key是整数的时候可以当O(1)因为整数就固定4个字节，
                        #                               而key是字符串的时候是不定长的(不知道多少个字节)只能用L代表长度
        return neighbors

    # 领口 E 433 Number of Islands 连通块问题
    def numIslands(self, grid):
        """
        这题虽然没有直接告诉是graph了，但有点像是求 有几个连通块的问题
        那就可以抽象化成 对graph做bfs的问题
        怎么抽象？
        二纬数组，变换成 坐标（这是个小技巧），坐标是有规律，从而找到每个坐标的邻居

        总的来讲，这个题是把每个点访问一遍，只有是岛的并且没访问过的才bfs
        所以最快情况，全是岛屿的话，时间空间复杂度O(MN), M和N是这个矩阵的长宽，当然我也不知道我分析得对不对

        """
        if not grid or not grid[0]:
            return 0

        islands = 0
        visited = set()
        '''
        矩阵坐标系 (0,0)
                    ----------> Y
                    |         x-1,y (-1,0)
                    | x,y-1     x,y    x, y+1
                    | (0,-1)  x+1,y     (0,1)
                  X |            (1,0)
            数组 变换 成坐标，由坐标上下左右的坐标，判断两座吧是否相邻
            所以DIRECTIONS = 【(1,0), (0,-1), (-1,0), (0,1)】
        '''
        rows = len(grid)
        columns = len(grid[0])
        for x in range(rows):
            for y in range(columns):
                # 总的来讲，时间复杂度该算这条句子的执行次数，这条语句执行了 MN 次，MN是矩阵的长宽
                if grid[x][y] and (x, y) not in visited:
                    # 如果grid[x][y]是个岛，并且(x,y) 是没访问过的，对这个坐标点做bfs
                    self.bfs120(grid, x, y, visited)
                    islands += 1

        return islands
    def bfs120(self, grid, x, y, visited):
        """
        这个 bfs 的时间复杂度应该是O(V), V是这个联通块的岛屿的个数。由于邻居固定是4个，所以不需要计算边
        """
        from collections import deque
        DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  #  这条语句写在class外就是一个全局常量

        queue = deque([(x, y)])
        visited.add((x, y))
        while queue:
            x, y = queue.popleft()
            neighbors = [(x+delta_x, y+delta_y) for (delta_x, delta_y) in DIRECTIONS]
            for neighbor in neighbors:
                if not self.is_valid(grid, neighbor[0], neighbor[1], visited):
                    continue                #     x         y
                queue.append(neighbor)
                visited.add(neighbor)
            '''
            # 以上7句也可以写成
            for delta_x, delta_y in DIRECTIONS:
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.is_valid(grid, next_x, next_y, visited):
                    continue
                queue.append((next_x, next_y))
                visited.add((next_x, next_y))
            '''
    def is_valid(self, grid, x, y, visited):
        # 这个函数是 O(1)
        rows, columns = len(grid), len(grid[0])
        # step 1 先确保 x y 是在范围内的
        if not (0 <= x < rows and 0 <= y < columns):
            return False
        # step 2 要确保 (x,y) 还没访问过
        if (x, y) in visited:
            return False
        # step 3 还要确保 grid[x][y] 是个岛
        return grid[x][y] == 1

    # 领扣 M 611 · Knight Shortest Path  最短路径  但做法类似领扣433
    def shortestPath(self, grid, source, destination):
        from collections import deque
        # 注意把数组转化成坐标的时候，用的是矩阵坐标系
        DIRECTIONS = [(1,2), (1,-2), (-1,2), (-1,-2), (2,1), (2,-1), (-2,1), (-2,-1)]

        que = deque( [(source.x, source.y)] )
        visited_distance = { (source.x, source.y): 0 }   # 不能直接吧source放进哈希表里，
                                                         # 因为source的类型是list，list/dict/set都是不可以哈希的
        while que:
            x, y = que.popleft()

            if (x,y) == (destination.x, destination.y):
                return visited_distance[(x,y)]

            for d_x, d_y in DIRECTIONS:
                next_x = x + d_x
                next_y = y + d_y
                if not self.isvalid(next_x, next_y, visited_distance, grid):
                    continue
                visited_distance[(next_x, next_y)] = visited_distance[(x, y)] + 1
                que.append((next_x, next_y))

        return -1
    def isvalid(self, x, y, visited_distance, grid):
        rows = len(grid)
        columns = len(grid[0])

        if not(0 <= x < rows and 0<= y < columns):
            return False

        if (x,y) in visited_distance:
            return False

        return grid[x][y] == 0

    # 领扣 M 127 · Topological Sorting 模版写法，背！！！ 注意 vertex: [node1,node2,node3] 指的是 vertex指向的node们
    def topSort(self, graph):
        """
        Time Complexity: O(V+E)
        当然，这个写法的前提条件是，整个graph里是没有cycle的，所以是一定有 topological sort 的

        一般如果没有这个前提条件的话，还是要确保graph是没有cycle才能有 topo_order
        那怎么判断有没有cycle呢？  最后 len(topo_order) == number_of_vertices 才会有 topological order
        具体可以看下616题
        """
        nodes_indegree = self.get_indegree(graph)

        # bfs
        order = []
        # 先找到入度为0的点，放入queue
        start_nodes = [n for n in graph if nodes_indegree[n] == 0]
        queue = collections.deque(start_nodes)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in node.neighbors:
                nodes_indegree[neighbor] -= 1
                # 如果 nodes_indegree[neighbor] 入度是0了就放入队列
                if nodes_indegree[neighbor] == 0:
                    queue.append(neighbor)

        return order
    def get_indegree(self, graph):
        nodes_indegree = {x: 0 for x in graph}

        for node in graph:
            for neighbor in node.neighbors:
                # 注意node指向它邻居，是邻居的in-degree来+1
                nodes_indegree[neighbor] += 1

        return nodes_indegree

    # 领扣 M 616 · Course Schedule II 是否存在拓扑排序, 存在的话就输出
    def findOrder(self, numCourses, prerequisites):
        """
        time/space complexity : O(V+E)
        """
        from collections import deque
        # step 1: 把先修课抽象成graph，建立graph和算出 in-degree
        graph, in_degree = self.build_graph892(numCourses, prerequisites)
        # step 2: bfs
        start_nodes = [course for course in in_degree if in_degree[course] == 0]
        queue = deque(start_nodes)
        topo_order = []

        while queue:
            course = queue.popleft()
            topo_order.append(course)

            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

        # 判断graph 有没有cycle：len(topo_order) == number_of_vertices
        if (len(topo_order) == numCourses):
            return topo_order

        return []
    def build_graph(self, numCourses, prerequisites):
        # 建图
        graph = {i: set() for i in range(numCourses)}
        for course, prerequisite in prerequisites:
            graph[prerequisite].add(course)

        # 算图里每个点的 in_degree (易错点，算入度跟建图分开搞，不要掺和在一个for循环里搞，因为有可能会有重复计算比如 [9,1] [9,1] 导致9的入度是2，但实际上只是1)
        in_degree = {i: 0 for i in range(numCourses)}
        for vertext in graph:
            hash_set = graph[vertext]
            for s in hash_set:
                in_degree[s] += 1

        return graph, in_degree

    # 领扣 M 605 · Sequence Reconstruction 总体上是 判断是否只存在唯一topological sort
    def sequenceReconstruction(self, org, seqs):
        """
        因为得是 subsequece of it, 还最短，所以必须是“前后”顺序紧紧依赖的感觉
        就让人想到 topological sort, 那就先建个graph的抽象模型吧(难度更大了)
        难点：
        1.根本没读懂题意的具体要求
        2.就算读懂，抽象成graph模型还是有点难想到（因为不是直接的前/后关系）
        3.就算想到了，在用什么建立graph的node时也错了
        4.就算写出来了cornor case也比较难照顾到，比如 org=[1] seqs=[[]]
        """
        from collections import deque

        graph, in_degree = self.build605(org, seqs)

        # 如果长度都不等，就可以直接返回了
        if len(graph) != len(org):
            return False

        # topological sort，bfs
        start_nodes = [node for node in in_degree if in_degree[node] == 0]
        topo_order = []

        queue = deque(start_nodes)  # 注意 start_node就已经是个list了
        while queue:
            # 检查 topological sort order 是否唯一的关键  队列里的size要么是0要么是1
            if len(queue) >= 2:
                return False

            node = queue.popleft()
            topo_order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return topo_order == org  # 这句话同时也检测了 拓扑排序是否存在
    def build605(self, org, seqs):
        # build  易错点: 这道题里要用 seqs 里的元素来建立graph模型，而不是用org
        graph = {}
        for seg in seqs:
            length = len(seg)
            if length == 0:
                continue
            for i, num in enumerate(seg):
                graph.setdefault(num, set())
                if 1 <= i < length:
                    graph[seg[i - 1]].add(num)

        # in-degree
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for s in graph[node]:
                in_degree[s] += 1

        return graph, in_degree

    # 领扣 H 892 · Alien Dictionary 求按字典最小的唯一拓扑序
    def alienOrder(self, words):
        """
        时间复杂度实在是分析不出来了
        这题难点：
        1. 题根本就没读懂
        2. 读懂后建图也超级容易出错
        3. 因为建图的时候还要考虑到用谁建node不会漏掉，如果建图用的顺序是in valid怎么办，如果 ['abc', 'ab'] 那么这样的顺序是 invalid 的
        4. 时间复杂度直接放弃
        """
        from heapq import heapify, heappop, heappush

        graph = self.build_graph892(words)

        if not graph:  # 易错点，edge case忘记写
            return ''

        indegrees = self.get_indegrees892(graph)

        priority_q = [node for node in indegrees if indegrees[node] == 0]
        heapify(priority_q)
        topo_order = []

        while priority_q:
            char = heappop(priority_q)
            topo_order.append(char)
            for neighbor in graph[char]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    heappush(priority_q, neighbor)

        # 易错点：不要漏掉这个条件是检测graph 有没有 cycle 的
        if len(topo_order) == len(graph):
            return ''.join(topo_order)

        return ''
    def build_graph892(self, words):
        graph = {}
        # build nodes:   易错点,如果同时 build nodes 和 edges比较绕，那就分开build
        for word in words:
            for c in word:
                if c not in graph:
                    graph[c] = set()

        # build edges:
        for i in range(1, len(words)):
            w1 = words[i - 1]
            w2 = words[i]
            size = min(len(w1), len(w2))
            for j in range(size):
                c1 = w1[j]
                c2 = w2[j]
                if c1 != c2:  # 说明 c1 -> c2
                    graph[c1].add(c2)
                    break

                # corner case: 易错点和难点  如果 ['abc', 'ab'] 那么这样的顺序是 invalid 的
                if j == (size - 1) and len(w1) > len(w2):
                    return None

        return graph
    def get_indegrees892(self, graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:  # for c in graph.keys():
            node_points_to = graph[node]
            for c in node_points_to:
                indegrees[c] += 1
        return indegrees

    # 领扣 M 788 · The Maze II 复杂图的最短路径，用 SPFA 来解决的（就是在简单图的BFS上作变形）
    def shortestDistance1(self, maze, start, destination) -> int:
        """
        1.就着example把跟着代码走一遍，好难啊
        2.除了层级，还要顾到走了多少步. 所以这不是一个简单图！！！不能用层级来算步数！！
        3.edges的weight不一样，算是复杂图，用 SPFA 解法
            把所有 nodes 的 step 设置为 无限大
            除非遇到 node 的 step 更小了，再更新这个 node 的 step
            （SPFA解决不了edge weight是负数的那种）

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

            while self.is_valid2(next_x + dx, next_y + dy, maze):  # 这里加一个在不在dict里的检查
                next_x += dx
                next_y += dy
                count_steps += 1

            if count_steps > 0:
                next_nodes[(next_x, next_y)] = pre_steps + count_steps

        return next_nodes
    def is_valid2(self, x, y, maze):
        # 没有撞墙也没有走出地图就是 valid
        if not (0 <= x < len(maze) and 0 <= y < len(maze[0])):
            return False

        if maze[x][y] == 1:
            return False

        return True
    #
    # 领扣 M 788 · The Maze II 复杂图的最短路径，把二维复杂图 转化成三纬 简单图（做BFS解决的） 这个写法比法1简单，但可能比较抽象
    def shortestDistance2(self, maze, start, destination) -> int:
        """时间复杂度是O(mn) 至少chris是这么说的，因为每个点会visit 5次(有5个directons)"""
        from collections import deque
        #               U       R      D      L
        DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # 其实除了4个明确方向，还有1个 None （就是需要重新选择方向的状态）

        # make "start" and "destination" as tuple cuz they are list at the beigining
        start = (start[0], start[1], None)
        destination = (destination[0], destination[1], None)

        # initialize
        queue = deque([start])
        visited = {start: 0}

        while queue:
            # current node info
            curr_node = queue.popleft()
            curr_x = curr_node[0]
            curr_y = curr_node[1]
            curr_direction = curr_node[2]

            # stopping case
            if curr_node == destination:
                return visited[curr_node]

            # case 1: next node should re-choose a direction
            if curr_direction == None:
                # choose next possible direction
                for next_direction in DIRECTIONS:
                    # calculate next node info
                    next_x = curr_x + next_direction[0]
                    next_y = curr_y + next_direction[1]
                    next_node = (next_x, next_y, next_direction)

                    if self.is_vaild3(maze, next_node):
                        self.process_next_node(next_node, curr_node, visited, queue)

            # case 2: next node will go along its current direction
            if curr_direction != None:
                # calculate next node info with current direction
                next_direction = curr_direction
                next_x = curr_x + next_direction[0]
                next_y = curr_y + next_direction[1]
                next_node = (next_x, next_y, next_direction)

                # case 2.1: if next_node is valid
                if self.is_vaild3(maze, next_node):
                    self.process_next_node(next_node, curr_node, visited, queue)
                else:
                    # case 2.2: if next_node is NOT valid, that means next node will meet a barrier or boundary
                    #           so we should go back the current node's x and y, but direction should set to None
                    next_node = (curr_x, curr_y, None) # assign next_node’s x value with current node‘s x
                    if next_node in visited:
                        continue
                    queue.append(next_node)
                    visited[next_node] = visited[curr_node]

        return -1
    def process_next_node(self, next_node, curr_node, visited, queue):
        if next_node in visited:
            return
        queue.append(next_node)
        visited[next_node] = visited[curr_node] + 1
    def is_vaild3(self, maze, node):
        n, m = len(maze), len(maze[0])
        x = node[0]
        y = node[1]

        if x < 0 or x >= n:
            return False
        if y < 0 or y >= m:
            return False

        return not maze[x][y]


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

if __name__ == '__main__':

    input = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]]
    sol = Solution()
    l = sol.shortestDistance2(input, [0,4],[4,4])

    print(l)

    pass