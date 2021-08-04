# 集合讲解
'''
集合的 添加，删除，查找，取长度是O(1)
遍历是 O(n)
remove元素，如果找不到会报错。还是discard比较安全

集合间的操作：都会生成新的集合
并集 ｜
交集 &
差 - （操作数前后顺序，不一样  ）
对称差：它俩并集-它俩交集

应用：
set插入和查找操作效率很高，O(1)
非常适合用来记录某元素，是否出现过
'''

class Solution:

    # lintcode Easy 487 · Name Deduplication
    def nameDeduplication(self, names):
        """直接修改原list，do it in-place 不需要额外的空间消耗"""
        s = set()
        j = 0
        for (i, name) in enumerate(names):

            if name.lower() not in s:
                s.add(name.lower())
                names[j] = name.lower()
                j += 1

        return names[0:j]

    # lintcode Easy 521 · Remove Duplicate Numbers in Array
    def deduplication(self, nums):
        exist = set()
        counter = 0
        for each in nums:
            if each not in exist:
                exist.add(each)
                nums[counter] = each
                counter += 1
        return counter

    # lintcode Easy 1796 · K-Difference
    def KDifference(self, nums, target):
        s = set()
        counter = 0
        for each in nums:
            candidate1 = target + each
            candidate2 = each - target
            if candidate1 in s:
                counter += 1
            if candidate2 in s:
                counter += 1
            s.add(each)
        return counter

if __name__ == '__main__':

    sol = Solution()
    l = sol.KDifference([1, 3, 5, 7],2)

