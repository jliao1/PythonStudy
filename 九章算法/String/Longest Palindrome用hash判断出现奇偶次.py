'''
Description
Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.

This is case sensitive, for example "Aa" is not considered a palindrome here.

总结：自己没做出来，因为把英文题又给看错题意了。是求组合，不是 substring 能不能组成 palindrome
'''
import collections


class Solution:
    """
    @param s: a string which consists of lowercase or uppercase letters
    @return: the length of the longest palindromes that can be built
    """
    def longestPalindrome1(self, s):
        # {} 是字典 key是字符，value是 True或False
        hash = {}

        for c in s:
            if c in hash:
                # 某字符，若已经出现过(在hash表里)，说明是偶数次了, 就删掉它
                del hash[c]
            else:
                # 某字符，若在hash表里，说明是奇数次, 就添加它
                hash[c] = True

        # 这是出现奇数次的字母个数
        remove = len(hash)

        # 在所有出现次数是奇数的字母中，留一个就好了
        if remove > 0:
            remove -= 1
                      # 其他删掉
        return len(s) - remove

    # 贪心算法
    def longestPalindrome2(self, s):

        #cnt统计字符串s中每种字母出现次数的计数数组
        #OddCount为是否有奇数次字符，1表示有，0表示无
        #ans为最终答案

        ans = 0
        cnt = collections.Counter(s)
        #每种字符可使用cnt/2*2次
        #如果遇到出现奇数次的字符并且中心位置空着，那么答案加1

        for i in cnt.values():
            # 如果出现了cnt次，那么我们可以将这个字符在字符串两侧各放置 cnt/2个
            ans += i // 2 * 2
            # 如果 ans 是偶数次，
            # 并且有字符出现奇数次，那么我们将其放在回文串中
            if ans % 2 == 0 and i % 2 == 1:

                ans += 1
        return ans



if __name__ == '__main__':
    sol = Solution()
    res = sol.longestPalindrome1("abccccdd")




