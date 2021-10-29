class Solution:
    # 力扣M 57 Insert Interval 这是我自己的解法，速度倒是挺快的，但代码看起来好长。看版本2令狐冲的答案(但是lintcode的答案)
    def insert(self, intervals, newInterval) :
        # edge case: newInterval 跟 intervals 没有相交, 那就直接插在后面返回
        if intervals == [] or intervals[-1][1] < newInterval[0]:
            intervals.append(newInterval)
            return intervals

        from collections import deque
        queue = deque()

        last_start = intervals[-1][0]
        last_end = intervals[-1][1]

        new_start = newInterval[0]
        new_end = newInterval[1]

        while intervals and new_end < last_start:
            temp = intervals.pop()
            queue.appendleft(temp)
            if intervals:
                last_start = intervals[-1][0]
                last_end = intervals[-1][1]

        if intervals:
            last_start = intervals[-1][0]
            last_end = intervals[-1][1]
        while intervals and last_end >= new_start:
            new_start = min(last_start, new_start)
            new_end = max(last_end, new_end)
            intervals.pop()
            if intervals:
                last_start = intervals[-1][0]
                last_end = intervals[-1][1]

        intervals.append([new_start, new_end])
        if queue:
            intervals.extend(list(queue))

        return intervals

    # 力扣M 57 Insert Interval 令狐冲写法，巧妙利用插入list某个位置元素，这样写出来就比较简单
    def insert(self, intervals, newInterval):
        results = []
        insertPos = 0

        new_start = newInterval[0]
        new_end = newInterval[1]

        for interval in intervals:
            internal_start = interval[0]
            internal_end = interval[1]

            if internal_end < new_start:
                results.append(interval)
                insertPos += 1
            elif internal_start > new_end:
                results.append(interval)
            else:
                new_start = min(internal_start, new_start)
                new_end = max(internal_end, new_end)
                #                                            List = [1, 2, 3]
        results.insert(insertPos, [new_start, new_end])   #  List.insert(1, 0)  # 插入之后是 [1, 0, 2, 3]
        return results

    # 力扣 56 Merge Intervals 令狐冲写法，不错，很清晰！
    def merge(self, intervals): #  intervals: List[List[int]]
        intervals.sort()
        result = []
        for interval in intervals:
            start = interval[0]
            end = interval[1]
            #                      last_end = result[-1][1]
            if len(result) == 0 or result[-1][1] < start:  # 易错点：这个地方的条件其实很不容易想透彻
                # 这种情况不会相交
                result.append(interval)
            else:
                # 我们只要改变 last_end 就好了，last_start是不用改变的，因为 last_start一定小于等于 start, 因为之前sort过了
                result[-1][1] = max(result[-1][1], end)

        return result

if __name__ == '__main__':
    sol = Solution()
    res = sol.insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8])
    print(res)

