class Solution:
    """
    @param words: the words
    @param S: the string
    @return: the string with least number of tags
    """

    # 自己写的版本，只找出了  allAprearenceIdx = [5, 7, 9, 11, 12, 15, 19] 该bold
    def boldWords1(self, words, S):
        # Write your code here
        allAprearenceIdx = []
        for each in words:
            self.helper(each, S, allAprearenceIdx, 0)

        # 排个序
        allAprearenceIdx.sort() # 这样要写在前面，因为set()对象不能sort
        # 转成set去重
        allAprearenceIdx = set(allAprearenceIdx)
        # 再把set转成list
        allAprearenceIdx = list(allAprearenceIdx) # 再变回list才能使用下标，set对象不能使用下标

        idx1 = 1
        # 这个list 是放 应该删除哪些元素
        list1 = []
        while(idx1+1 < len(allAprearenceIdx)):
            if allAprearenceIdx[idx1] - allAprearenceIdx[idx1-1] == 1 and allAprearenceIdx[idx1+1] - allAprearenceIdx[idx1] == 1 :
                list1.append(idx1)
                idx1 +=1
            else:
                idx1 += 1
        for i in range(len(list1)-1,-1,-1):
            allAprearenceIdx.pop(list1[i])

        tag = '<b>'
        S = list(S)
        for i in range(len(allAprearenceIdx)-1,-1,-1):
            S.insert(allAprearenceIdx[i], tag)

        return allAprearenceIdx



    def helper(self, each, S, allAprearenceIdx, start):
        if each in S[start:]:
            firstIdx = S[start:].index(each) + start
            for i in range (firstIdx, firstIdx + len(each)):
                allAprearenceIdx.append(i)

            self.helper(each, S, allAprearenceIdx, firstIdx + 1)

        else:
            return


    # 精选答案，思路是一样的，但这个版本熟练用python
    def boldWords2(self, words, S):
        # Write your code here
        bold = set()
        for w in words:
            i, end = S.find_target_node_and_its_parent(w, 0), -1
            # i >= 0 说明在S里找到w了，否则 没找到
            while i >= 0:
                # 这句是 把 range 里的 数字 都加到 bold 里
                bold.update(range(max(i, end + 1), i + len(w)))
                end = i + len(w) - 1
                # 从 i+1 位开始，接着 继续找 w
                i = S.find_target_node_and_its_parent(w, i + 1)   # find里，第一个参数是要寻找的sub，第二个是从哪里开始找，第三个参数是找到哪位之前

        # 代码走到这里，也就得出了 应该加粗 S 的哪些 index
        # 下面是我没做出来的部分（理解不到位）， in 和 不in 有什么区别

        res = []
        for i in range(len(S)):
            # 当它需要加粗，它前面那位不需要加粗时，插个tag
            if i in bold and i - 1 not in bold:
                res.append('<b>')

            res.append(S[i])

            # 当它需要加粗，它后那位不需要加粗时，插个tag
            if i in bold and i + 1 not in bold:
                res.append('<b>')

        return ''.join(res)


if __name__ == '__main__':
    sol = Solution()
    words1 = ["bcccaeb", "b", "eedcbda", "aeebebebd", "ccd", "eabbbdcde", "deaaea", "aea", "accebbb", "d"]
    S1 = "ceaaabbbedabbecbcced"

    words2 = ["ab", "bc"]
    S2 ="aabcd"


    res = sol.boldWords2(words1, S1)
    print(res)

