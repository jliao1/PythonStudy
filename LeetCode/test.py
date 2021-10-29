# # return： false true
#
# # 题目 https://leetcode.com/discuss/interview-question/406659/Twitter-or-OA-2019-or-Get-Set-Go
# import heapq
#
#
# def isPossible(input, goal):
#     if not input:
#         return False
#
#     input.sort()
#     return dfs(input, goal, 0)
# def dfs(input, goal, index):
#     # stopping case
#     if goal == 0:
#         return True
#
#     for i in range(index, len(input)):
#         curr = input[i]
#         temp_goal = goal - curr
#         if temp_goal < 0:
#             break
#         if dfs(input, temp_goal, i + 1):
#             return True
#
#     return False


class Solution:
    def maxProbability(self, n: int, edges, succProb, start, end) :
        from heapq import  heappop, heappush
        graph, prob = dict(), dict()  # graph with prob
        for i, (u, v) in enumerate(edges):
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)
            prob[u, v] = prob[v, u] = succProb[i]

        #  这个h是只时刻追踪当前这一轮的所有点
        h = [(-1, start)]  # Dijkstra's algo
        processed = set()       # 已经process好的点
        while h:
            p, n = heappop(h)
            if n == end: return -p
            processed.add(n)
            for nn in graph.get(n, []):
                if nn in processed: continue
                temp = (p * prob.get((n, nn), 0), nn)
                heappush(h, temp)
        return 0

    Direction = [(-1, 0), (0, 1), (-1, 0), (0, -1)]

    def shortestDistance(self, maze, start, destination):
        from heapq import heapify, heappop, heappush  # priority queue
        # dijkstra
        #    element: [distance_to_start_point, position] [(0，4)，0]
        q = [(0, start)]
        processed = set()

        while q:
            distance, [x, y]= heappop(q)
            if [x, y] == destination:
                return distance

            if (x, y) in processed:
                continue

            processed.add((x, y))  # 可能弹出的xy已经process过的，需要continue掉

            # get possible_next_steps

            possible_next_positions = self.get(maze, x, y)
            for steps, position in possible_next_positions:
                if position in processed:
                    continue
                heappush(q, (distance + steps, position))

        return -1

    def get(self, maze, x, y):
        Directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        possible_next_steps = []

        for dx, dy in Directions:
            next_x = x
            next_y = y
            steps = 0
            while self.is_valid(next_x + dx, next_y + dy, maze):
                next_x += dx
                next_y += dy
                steps += 1
            if steps!= 0:
                possible_next_steps.append( (steps, (next_x,next_y)))

        return possible_next_steps

    def is_valid(self, x, y, maze):
        # within range
        if x < 0 or x >= len(maze):
            return False
        if y < 0 or y >= len(maze[0]):
            return False

        # not a wall
        return maze[x][y] != 1

if __name__ == '__main__':
    m = {1:11,2:22}
    v = [m[key] for key in m]
    a = d.get(3)
    n = 3
    edges = [[0, 1], [1, 2], [0, 2]]
    succProb = [0.5, 0.5, 0.8]
    start = 0
    end = 2

    maze = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    start = [0,4]
    destination = [4, 4]

    sol = Solution()
    res = sol.shortestDistance(maze, start, destination)
    print(res)

    sol = Solution()
    res = sol.maxProbability(3, [[0,1],[1,2],[0,2]], [0.5,0.5,0.2], 0, 2)
    print(res)
