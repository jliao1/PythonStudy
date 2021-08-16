#!/bin/python3

import math
import os
import random
import re
import sys

"""
第一题 Global maximum 题目 https://leetcode.com/discuss/interview-question/1215681/airbnb-oa-global-maximum
第一题解答 https://www.geeksforgeeks.org/maximum-of-minimum-difference-of-all-pairs-from-subsequences-of-given-size/
"""
# Complete the 'findMaximum' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY arr
#  2. INTEGER m
#

def findMaximum(A, m):
    n = len(A)
    A.sort()

    start = 0
    end = A[n - 1] - A[0]

    res = 0

    while start <= end:
        mid = (start + end) // 2
        if can_get_a_sub_of_size_m(A, n, m, mid):
            res = mid
            # find a better answer in the right half
            start = mid + 1
        else:
            end = mid - 1

    return res

def can_get_a_sub_of_size_m(A, n, m, mid):
    count = 1
    last = A[0]
    for i in range(1, n):
        if A[i] - last >= mid:
            last = A[i]
            count += 1

            if (count == m):
                return True
    return False

# 上面是题目原解，下面是自己加工的

# 四重境界 OA题 Global maximum 题目 https://leetcode.com/discuss/interview-question/1215681/airbnb-oa-global-maximum
def findMaximum2(self, A, K):
    """时间复杂度O[Nlog(end)]"""
    n = len(A)
    A.sort()

    # 注意这个start和end是答案范围
    start = 0
    end = A[n - 1] - A[0]

    while start + 1 < end:
        mid = (start + end) // 2
        # check whether it is possible to get a subString of size K with a minimum difference among any pair equal to mid.
        # 是否存在一个是 K size 的 sub_array, 这个 sub_array 里 两两的差，最小是 mid
        is_exist, _ = self.exist_sub_of_size_k(A, K, mid)
        if is_exist:
            # 如果存在
            # 去right half 找，因为这题要求的 mid 尽量大
            start = mid
        else:
            # 去left half 找
            end = mid

    is_exist, _ = self.exist_sub_of_size_k(A, K, end)
    if is_exist:
        return end

    is_exist, _ = self.exist_sub_of_size_k(A, K, start)
    if is_exist:
        return start

    return res
def sub_of_size_k(self, array, K, min_difference):
    """
    看 array 里 是否存在一个 sub-array，它的 minimum difference among any pair equal to "min_difference"
    这个 sub-array 里的元素的两两之差, 最小的差是 "min_difference"
    如果是的话，返回 true，并返回len 最短的这个sub-array
    时间复杂度O(n)
    """
    # 注意, 传进来的array已经是sort好的了

    len_of_sub = 1
    sub = [array[0]]
    last = array[0]

    for i in range(1, len(array)):  # i 从第二个数开始
        '''
        因为 array 已经是 sorted 的从 小->大的
        在pick元素的时候，只要 这个array[i] 与 last 之差大于 min_difference
        就能保证 shortest_sub 里俩俩之差，最小的是 min_difference 啦
        '''
        if array[i] - last >= min_difference:
            # 说明 array[i] 符合要求，pick 它!
            sub.append(array[i])
            # 把last更新一下
            last = array[i]
            len_of_sub += 1
            # 每pick完一个元素，检查一下是否符合想要的K长度，是的话就立刻返回, 这样才能保证找到符合要求的sub是shortest的
            if (len_of_sub == K):
                print("shortest_sub:", sub)
                return True, sub
    return False, -1

if __name__ == '__main__':
    A = [1,2,3,4]
    m = 3
    res = findMaximum(A, m)
    print(res)