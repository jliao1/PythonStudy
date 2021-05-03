# Leetcode: Palindrome Data Stream

class Solution:
    """
    @param s: The data stream
    @return: Return the judgement stream
    """

    def getStream(self, s):
        # 先处理 edge case
        if s is None:
            return []

        res = []

        # 需要数出现次数为奇数的字母，那就需要一个字母表 (需要有一个 对每个字母出现次数的 计数)
        alphabet = [0] * 26  # 开一个长度为26的初始值为0的这么一个list
        # index 0 代表a，1代表b…… 25代表z

        count = 0  # 还需要一个计数
        for i in range(len(s)):  # 开始读了
            # 记录当前扫到的 每一个字母, 目前出现了多少次
            alphabet[ord(s[i]) - ord('a')] += 1

            # 如果出现的次数已经为奇数, count +1
            if alphabet[ord(s[i]) - ord('a')] % 2 == 1:
                count += 1
            else: # 如果是偶数, count -1
                count -= 1

            # 利用回文串性质，如果 出现次数为奇书的 字母个数<=1才是回文串，不然就不是汇文串
            if count > 1:
                res.append(0)
            else:
                res.append(1)
        return res

if __name__ == '__main__':
    sol = Solution()



'''
知识点：
回文串的性质： 出现次数为奇书的字母的个数 <=1
'''