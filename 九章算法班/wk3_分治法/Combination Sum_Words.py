class Solution:
    # 亚麻internVO题 力扣39 M Combination Sum 这是一种暴力DFS, 很慢，看写法2会好点
    def combinationSum1(self, candidates, target: int) :
        # dfs
        if not candidates:
            return []

        result = set()

        self.dfs_combinationSum1(candidates, target, result, [])

        return list(result)
    def dfs_combinationSum1(self, candidates, target, result, combination):
        if target < 0:
            return

        if target == 0:
            combination.sort()
            # 去重
            if tuple(combination) in result:
                return
            else:
                result.add(tuple(combination))
                return

        for i in range(len(candidates)):
            curr = candidates[i]
            temp_target = target - curr

            temp_combination = list(combination)
            temp_combination.append(curr)
            self.dfs_combinationSum1(candidates, temp_target, result, temp_combination)

    # 亚麻internVO题 力扣M 39 Combination Sum 这是一种 DFS + 3sum排序去重加快速度，这题有意思
    def combinationSum2(self, candidates, target):
        """
        例子 求可以组成7的数字组合，可以用candidates里的重复元素
        Input: candidates = [2,3,6,7], target = 7
        Output: [[2,2,3],[7]]
        """
        results = []

        # 集合为空
        if len(candidates) == 0:
            return results

        # 利用set去重，再排序  反正 相同index的数据也可以取多次，那就在库里就存1个就好
        #                技巧！！！这种先排序可以去（1）按顺序依次处理，避免走回头路（2）加快速度
        candidatesNew = sorted(list(set(candidates)))

        # dfs
        self.dfs_combinationSum2(candidatesNew, 0, [], target, results)

        return results
    def dfs_combinationSum2(self, candidates, index, combination, remainTarget, results):
        # 到达边界
        if remainTarget == 0:
            return results.append(list(combination))  # 要deepcopy一下加进去，不然往后combination会一直变

        # 递归的拆解：挑一个数放入current
        for i in range(index, len(candidates)):
            curr = candidates[i]

            # 剪枝 (remainTarget-curr是负数的话，就不用往下找了，这是一个加速过程)
            #      curr之前的也不用回头看了，因为是处理排序后的数组，去重了
            if remainTarget - curr < 0:
                break  # 是接下来的元素，都不执行了

            combination.append(curr)      # 注意传入的依然是 i 而不是i+1 因为这题可以重复利用相同元素
            self.dfs_combinationSum2(candidates, i, combination, remainTarget - curr, results)
            # 技巧！！！combination加完了，处理完后，再把元素pop出来
            combination.pop()

    # 力扣39 Combination Sum的变种题，这是推特oa题：https://leetcode.com/discuss/interview-question/406659/Twitter-or-OA-2019-or-Get-Set-Go
    def isPossible(self, input, goal):
        """
        例子：
        input = calCounts: {2,9,5,1,6}, requiredCals:12 Calories 挑选一些天吃的卡路里刚好等于12，不可以重复挑选
        output = True  because he can eat on days 0,1,3 (2+9+1=12) or on days 2,3,4 (5+1+6)


        """
        if not input:
            return False

        input.sort()
        return self.dfs_isPossible(input, goal, 0)
    def dfs_isPossible(self, input, goal, index):
        # stopping case
        if goal == 0:
            return True

        for i in range(index, len(input)):
            curr = input[i]
            temp_goal = goal - curr
            if temp_goal < 0:  # 减枝，不用往下了
                break
            if self.dfs_isPossible(input, temp_goal, i + 1):
                return True

        return False

    # 力扣MH 79. Word Search  一个word能不能在 matrix 里合成  这种写法外观看最简单
    def exist1(self, board, word):
        """
        Input: board = [["A","B","C","E"],
                        ["S","F","C","S"],
                        ["A","D","E","E"]], word = "ABCCED"
        Output: true
        空间复杂度 O(L)       L 是word的长度
        时间复杂度 O(N*4^L)   N是矩阵里的元素个数，4因为要走4个方向
        """
        for x in range(len(board)):
            for y in range(len(board[0])):
                if self.dfs_exist1(board, word, set([]), 0, x, y):
                    return True
        return False
    def dfs_exist1(self, board, word, visited, index, x, y):
        if index >= len(word):
            return True

        # 如果 越界||不是我们想找的数字||已经访问过, 就return False
        if (not 0 <= x < len(board)) or (not 0 <= y < len(board[0])) or \
                board[x][y] != word[index] or (x, y) in visited:
            return False

        visited.add((x, y))

        Direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in Direction:
            next_x, next_y = x + dx, y + dy
            if self.dfs_exist1(board, word, visited, index + 1, next_x, next_y):
                return True
        visited.discard((x, y))
        return False

    # 力扣H 472 Concatenated Words 这是我用纯DFS做的，会stack overflow
    def findAllConcatenatedWordsInADict1(self, words):
        Map = {}
        for string in words:
            length = len(string)
            if length not in Map:
                Map[length] = set()
            Map[length].add(string)

        result = []

        for target in words:
            len_of_target = len(target)
            fileds = []
            for key, value in Map.items():
                if key < len_of_target:
                    fileds.extend(list(value))

            if len(fileds) < 2:
                continue

            if self.dfs472(target, fileds, ''):
                result.append(target)

        return result
    def dfs472(self, target, fields, current_string):
        if current_string == target:
            return True

        if current_string not in target:
            return False

        for string in fields:
            if self.dfs472(target, fields, current_string + string):
                return True

        return False

    def wordBreak(self, word, cands):
        if not cands:
            return False
        dp = [False] * (len(word) + 1)  # store whether w.substr(0, i) can be formed by existing words
        dp[0] = True  # empty string is always valid
        for i in range(1, len(word) + 1):
            for j in reversed(range(0, i)):
                if not dp[j]:
                    continue
                if word[j:i] in cands:
                    dp[i] = True
                    break
        return dp[-1]
    def findAllConcatenatedWordsInADict(self, words):
        words.sort(key=lambda x: -len(x))  # 排序，长度长的，排在前面
        cands = set(words)  # using hash for acceleration
        ans = []
        for i in range(0, len(words)):
            word = words[i]
            cands.discard(word)
            if self.wordBreak(word, cands):
                ans += word,
        return ans

if __name__ == '__main__':
    sol = Solution()
    res = sol.findAllConcatenatedWordsInADict(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])
    print(res)