# Leetcode: Palindrome Data Stream
# 题意理解：There is a data stream coming in, one lowercase
#          letter at a time. Can the arrangement of the
#          current data stream form palindrome string？
#          关键词是 "form"，比如给你 abab 虽然本身不是 palindrome，
#          但它是可以组成 回文串 的就可以返回1
'''
思路：
（1）常规模拟人做的思路是
        拿到所有的字母，然后求出它们可以组成所有的 permutations, 然后看这些permutations是否可以组成palindrome，可以的话返回1。
        当然这种做法挺慢的，有n个字母的话，可以组成 n! 种permutations
（2）利用 pailindorme 的奇偶性质解： 最多有一个字母出现奇数次，其他字母都出现偶数次。

知识点：
（1）处理异常/edge case的时候，通常从数据类型方向来思考，比如
        string 是个 object，那么 object 有可能就是个 None，然后string类型 的长度也有可能是 0
        再比如，若数据类型是 int，那么 int 有可能是 负数，是0，是超过出了max和min的范围

（2）快速开数组/开 list 的写法
    比如快速生成含有 26个0 的list：list = [0 for _ in range(26)]
                       或者写成：list = [0] * 26

（3）分类统计，用数组list，每个元素代表一类，用++来统计这个index的元素出现多少类

 (4) Ternary operator
     Java result[i] = odd_letter_cnt > 1 ?  0 : 1 ;
     转换成 python 是 result[i] = 0 if odd_letter_cnt > 1 else 1
'''

class Solution:
    """
    @param s: The data stream
    @return: Return the judgement stream
    """
    # 九章老师写法
    def getStream1(self, s):

        # 写法一：九章老师写法
        # 先处理 edge case
        # 处理异常可以从 数据类型 方向来思考
        if s is None or len(s) == 0:
            return []

        # 统计 字母出现奇数次 的字母的 个数
        odd_letter_cnt = 0

        # 目的是生成含有 len(s) 个 0 的 list：[0, 0, 0, 0, 0]
        result = [0 for _ in range(len(s))]

        # 目的是生成含有 26 个 0 的 list：[0, 0, 0, 0, 0]
        # 这个 letters 是(用来统计分类计数中的一个映射关系)，每个字母出现了几次
        letters = [0 for _ in range(26)]

        for i in range(len(s)):
            # 统计 index = ord(s[i]) - ord('a') 时，这个 letters[index] (映射的字母) 出现几次
            letters[ord(s[i]) - ord('a')] += 1
            # 如果 index 映射的字母，出现奇数次了
            if letters[ord(s[i]) - ord('a')] % 2 == 1:
                # odd_letter_cnt就加1
                odd_letter_cnt += 1
            else: # 说明当前字母一共出现了偶数次，odd_letter_cnt减1
                odd_letter_cnt -= 1

            # 利用回文串性质，如果 出现次数为奇书的 字母个数<=1才是回文串，不然就不是汇文串
            # Java的写法是 result[i] = odd_letter_cnt > 1 ?  0 : 1 ;
            result[i] = 0 if odd_letter_cnt > 1 else 1

        return result

    # 另一种写法
    def getStream2(self, s):
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


    '''
    这是我写的，但写错了
    这个题目要仔细审题哦，我们的题目是需要你判断数据流的排列是否能组成回文串，不是当前是否是回文串
    说到底还是英文不够好：
    There is a data stream coming in, one lowercase letter at a time. 
    Can the arrangement of the current data stream form palindrome string？
    '''
    def getStream3(self, s):
        # 先处理 edge case
        if s is None or s == '':
            return []

        res = [0] * len(s)
        res[0] = 1
        for i in range(1, len(s)):
            if self.helper(s, i) == True:
                res[i] = 1
            else:
                res[i] = 0
        return res

    def helper(self, s, idx):
        mid = idx // 2
        for i in range(0, mid+1):
            if s[i] != s[idx - i]:
                return False
        return True

if __name__ == '__main__':
    sol = Solution()
    s = 'kayak'
    num = sol.getStream2(s)
    print(num)
    print(s.find('ak'))



