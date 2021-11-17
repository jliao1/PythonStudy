# # Currency Conversion
# # We are given a list of currency conversions
#
# # USD →  EUR: 1.15   1 USD = 1.15 EUR    1 EUR = 1/1.15 USD
#
# # USD →  INR: 60     1 USD = 60 INR
#
# # EUR →  INR: 75
#
# # EUR →  YEN: 20
#
# # Having this list available, implement a currency converter which will take in 2 currency strings as input and provide a currency conversion output
#
# # for example
#
# # USD →  YEN: ?
#
#
# # rate
#
# # restes =[ ['USD', 'EUR', 1.15], ['USD', 'INR', 60], ['EUR','INR', 75] , ['EUR','YEN', 20]  ]
# # to_from = ['USD', 'YEN']
# # return float
#
#
# def graph(r):
#     g = {}
#     for currency in r:
#         fro = currency[1]  # USD: (eur, 1.15), (INR, 60)
#         to = currency[0]  # RUT: (usd, 1/1.15)
#         # INR: (USD, 1/60 ). (EUR, 1/75)
#         # EUR: (INR, 75), (YEN, 20)
#         # YEN: (EUR, 1/20)
#
#         g.setdefault(fro, set())
#         g.setdefault(to, set())
#
#         g[fro].add((to, 1 / currency[2]))
#         g[to].add((fro, currency[2]))
#
#     return g
#
#
# def func(r, tf):
#     g = graph(r)
#     print(g)
#     from collections import deque
#
#     start = tf[0]
#     end = tf[1]
#
#     queue = deque([start])  # queue : USD  EUR  INR
#     visited = {start: 1}  # value: how many value from USD to Key
#     # visited:  USD:1  EUR:1.15  INR:60   YEN:value
#     while queue:
#         node = queue.popleft()
#         if node == end:
#             return visited[node]
#
#         next_nodes = g[node]  # next_nodes = (eur, 1.15), (INR, 60)
#         # next_nodes =  (INR, 75), (YEN, 20)
#         for next_node, currency in next_nodes:
#             if next_node in visited:
#                 continue
#                 #     1         1.15
#             visited[next_node] = visited[node] * currency
#             queue.append(next_node)
#
#     return -1
#
#
# input1 = [['USD', 'EUR', 1.15], ['USD', 'INR', 60], ['EUR', 'INR', 75], ['EUR', 'YEN', 20]]
# input2 = ['USD', 'YEN']
#
# res = func(input1, input2)
# print(res)
#

class Solution:
    # 建图然后BFS  https://leetcode.com/discuss/interview-question/483660/google-phone-currency-conversion
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