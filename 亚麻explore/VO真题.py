class Solution:
    # 建图然后BFS  https: // leetcode.com / discuss / interview - question / 483660 / google - phone - currency - conversion
    def func(self, r, tf):
        g = self.graph(r)
        '''
         USD: (JPY, 110)
         JPY: (USD, 1/110) (GBP, 0.0070)
         US:  (AUD, 1.45)
         AUD: (US,  1/1.45)
      -> GBP: (JPY, 1/0.007)   1GBP = 1/0.007 JPY
        '''

        from collections import deque

        start = tf[0]
        end = tf[1]

        queue = deque([start])
        visited = {start: 1}

        while queue:
            node = queue.popleft()  # GBP
            if node == end:
                return visited[node]

            neighbors = g[node]
            for neighbor, num in neighbors:
                # JPY   0.007
                if neighbor in visited:
                    continue

                visited[neighbor] = visited[node] * num  # 1GBP = 1/0.007 JPY   JPY: 1/0.007
                queue.append(neighbor)

        return -1
    def graph(self, r):
        g = {}
        for currency in r:
            g.setdefault(currency[0], set())
            g.setdefault(currency[1], set())

            g[currency[0]].add((currency[1], currency[2]))
            g[currency[1]].add((currency[0], 1 / currency[2]))

        return g

if __name__ == '__main__':
    s = 'bckf'
    print(sorted(s))
    input1 = [ ['USD', 'JPY', 110], ['USD', 'AUD', 1.45], ['JPY', 'GBP', 0.0070] ]
    input2 = ['GBP', 'AUD']
    sol = Solution()
    res = sol.func(input1, input2)

    print(res)