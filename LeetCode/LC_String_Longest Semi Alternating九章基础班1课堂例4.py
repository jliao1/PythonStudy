# LeetCode：Longest Semi Alternating Substring 05.02.2021
# 感悟：双指针，提炼解题的关键是 “it does not contain three identical consecutive characters”

# 思路
# （1）brute force  求出所有的子串, 找里面是否有连续相同的3个字母,
#      找出没有的 连续相同3个字母的 长度最长的 那一个 （从长往短看，如果长的找到了，短的就不用看）
# （2）联想到更有技巧的思路 用双指针来做

'''
知识点：
（1）两个指针，一般用来解决字符串匹配问题。
     一前一后滑动，框定一个窗口。计算窗口中元素的问题（让元素合法）。
     用怎样的逻辑控制指针的移动，是题目重点之一
（2）一般题目里说 ，不能出现连续3个相等的 char，你就要联想到
    面试官 一会儿 可能要问你，如果不能出现连续 k 个相等的char，
    该怎么搞？
    当你开始考虑3个的时候，就要去考虑 k 个，因为一般follow up questions
    不需要把 code 重新推翻写一遍，一个好的代码只要在原基础上修改一点就好了。
    比如一般计算two sum，就会让你计算 three sum，four sum
'''

class Solution:
    """
    @param s: the string
    @return: length of longest semi alternating substring
    """

    # 九章老师讲课
    def longestSemiAlternatingSubstring(self, s):
        # 比如  "baaabbabbb"
        # 双指针  | |

        k = 3       # 不能 contain k 个 identical consecutive chars, 本题k是3
        n = len(s)  # 字符串长度

        # edge case1：先处理异常
        if s is None or n == 0:
            return 0

        # edge case2： 如果 输入的字符串s 长度小于3
        if n < 3:
            return n

        currMaxLen = 1  # 或者取名叫 res 也行
        cnt = 1  # 由于 doesn't contain three identical consecutive characters
                 # 所以需要一个 cnt 来 count 最后一个char(也就是right指向的char) 连续出现的次数，当 >= 3 就不合法了
                 # 由于 left 从0开始，right 从1开始，所以 cnt 和 currMaxLen 初始都是1

        left = 0  # 需要一个左指针，指向从 index = 0 开始
        # for 的是 right右指针，right 从 index = 1 开始，正常来讲，right走到 s 结尾就停止啦。但是可以做个小优化，如果s剩余的 位数已经 <= currMaxLen了也就没必要往下扫了
        for right in range(1, n):
            # 先判断 right 是否等于 right 前一位
            if s[right] == s[right - 1]:
                # 若相等,说明多出现了1个连续相等char
                cnt += 1
                # 如果已经连续k个char相等了，left～right之间框定的窗口就不合法了
                if cnt == k:
                    # 那就让窗口再次合法，那就把 left 移到 right 的前一位
                    left = right - (k-2)

                    # # 以下3行代码是个小优化，不写也行
                    remainingLen = n - left
                    if remainingLen <= currLen:
                        return currMaxLen

                    # 此时，由于left在right前一位，count该更新成 k-1（因为目前还是有2个char连续相等的）
                    cnt = k - 1
            else:  # right 不等于  right 向后移动一位
                cnt = 1  # 重新把cnt更新成1 （因为连续没有一个char跟 right 指向的char相等，所以cnt是1）

            currLen = right - left + 1     # 因为left只在count等于3时，才移到right前一位，所以这样是可以计算满足规则内的当前长度的
            currMaxLen = max(currMaxLen, currLen)  # 存下当前最大的 len

        return currMaxLen

if __name__ == '__main__':
    sol = Solution()
    res = sol.longestSemiAlternatingSubstring('baaabaaabbb')
    print(res)