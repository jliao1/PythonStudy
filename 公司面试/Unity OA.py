class Solution:
    # 题目 https://stackoverflow.com/questions/69005187/codility-stick-cut-challenge
    def solution1(self,A,B):

        a = A // 4;

        b = min(A, B // 3)

        c = min(A // 2, B // 2)

        d = min(A // 3, B)

        e = B // 4

        return max(a, max(b, max(c, max(d, e))))

    def solution2(self, N, A, B):
        if not A or len(A)<=1:
            return False

        graph = {}
        for i in range(N):
            graph.setdefault(i+1,set())

        for i in range(len(A)):
            a = A[i]
            b = B[i]
            graph[a].add(b)
            graph[b].add(a)

        # N = 4
        # 1 2 3 4
        for route in range(1,N):
            next_routes = graph.get(route, None)
            if next_routes is not None:
                if (route+1) not in next_routes:
                    return False
            else:
                return False

        return True




if __name__ == '__main__':
    sol = Solution()
    A = [1,2,4,4,3]
    B = [2,3,1,3,1]
    res = sol.solution2(4,A,B)
    print(res)


    ans = sol.solution1(4,0)
    print(ans)