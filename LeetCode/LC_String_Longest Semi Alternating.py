# LeetCode：Longest Semi Alternating Substring 05.02.2021
# 感悟：双指针，提炼解题的关键是 “it does not contain three identical consecutive characters”

class Solution:
    """
    @param s: the string
    @return: length of longest semi alternating substring
    """

    def longestSemiAlternatingSubstring(self, s):
        # 比如  "baaabbabbb"
        # 双指针  |

        # 先处理异常
        if s is None or len(s) == 0:
            return 0

        maxLen = 1  # 或者取名叫 res 也行
        left = 0  # 需要一个左指针
        count = 1  # 由于 doesn't contain three identical consecutive characters
        # 所以需要一个 count 来记录连续多少个字母相同了
        # 由于是从0和1开始的，所以count初始和maxLen初始都是1

        # for 的是 right，因为右指针到 s 结尾就停止啦
        for right in range(1, len(s)):
            #  先判断 right 是否等于 right 前一位
            if s[right] == s[right - 1]:
                count += 1  # 若相等,说明多出现了1次相等character
                if count == 3:  # 如果连续3个相等了
                    left = right - 1  # 那就把 左指针移到右指针前一位
                    count = 2  # 此时left在right左一位，count该更新成2
            else:  # right 不等于 right前一位
                count = 1  # 重新把count更新成1
            currLen = right - left + 1     # 因为left只在count等于3时，才移到right前一位，所以这样是可以计算满足规则内的当前长度的
            maxLen = max(maxLen, currLen)  # 存下当前最大的 len

        return maxLen

if __name__ == '__main__':
    sol = Solution()
    res = sol.longestSemiAlternatingSubstring('abaaaa')
    print(res)