
class Solution:
    def alienOrder(self, words):
        graph = self.build_graph(words)

        if not graph:  # 易错点，edge case忘记写
            return ''

        indegrees = self.get_indegrees(graph)

        from heapq import heapify, heappop, heappush

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

        # 易错点：这个条件是检测graph 有没有 cycle 的
        if len(topo_order) == len(graph):
            return ''.join(topo_order)

        return ''

    def build_graph(self, words):
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

        return graph

    def get_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:  # for c in graph.keys():
            node_points_to = graph[node]
            for c in node_points_to:
                indegrees[c] += 1
        return indegrees

if __name__ == '__main__':


    Input = ["z","x","z"]
    sol = Solution()
    l = sol.alienOrder(Input)
    print(l)
