'''
Description
We had some 2-dimensional coordinates, like "(1, 3)" or "(2, 0.5)". Then, we removed all commas, decimal points, and spaces, and ended up with the string S. Return a list of strings representing all possibilities for what our original coordinates could have been.

Our original representation never had extraneous zeroes, so we never started with numbers like "00", "0.0", "0.00", "1.0", "001", "00.01", or any other number that can be represented with less digits. Also, a decimal point within a number never occurs without at least one digit occuring before it, so we never started with numbers like ".1".

The final answer list can be returned in any order. Also note that all coordinates in the final answer have exactly one space between them (occurring after the comma.)
'''


class Solution:
    """
    @param S: An string
    @return: An string
    """
    # 这是自己写的，好像写错了，放弃
    def ambiguousCoordinates1(self, s):
        if s is None or len(s) == 0:
            return []
        res = []
        for i in range(1, len(s) - 2):
            part1 = s[:i]
            part2 = s[i:]
            list1 = self.helper(part1)
            list2 = self.helper(part2)
            for num1 in list1:
                for num2 in list2:
                    res.append(''.join(['(', num1, ', ', num2, ')']) )
        return res

    def helper(self, s):
        list = []
        if len(s) == 1:
            list.append(s)
            return list
        for i in range(1, len(s)):
            list.append(''.join([s[:i], '.', s[i:]]) )

        return list

    # 九章答案
    def ambiguousCoordinates2(self, S):
        ans = []
        # 去掉前后括号, 保留 第1位 ～ 倒数第1位(不包括倒数第1位)
        S = S[1:-1]
        for i in range(1, len(S)):
            # For each place to put the comma,
            # we separate the string into two fragments.
            left, right = S[:i], S[i:]

            left_list = self.get_number(left)
            right_list = self.get_number(right)

            if left_list and right_list: # 如果 left 和 right list 存在的话
                for left_number in left_list:
                    for right_number in right_list:
                        ans.append( ''.join(['(', left_number, ', ', right_number, ')']) )
            return ans


    # for each fragment, we have a choice of where to put the period
    def get_number(self, num):
        decimal_list = []
        if len(num) == 1 or num[0] != '0':
            decimal_list.append(num)
        for i in range(1, len(num)):
            # 整数    小数fractor 部分
            integer, fractor = num[:i], num[i:]
                #   if里第1个条件: 整数部分开头有0         if里第二个条件：小数部分最后有0
            if (len(integer) > 1 and integer[0] == '0') or fractor[-1] == '0':
                # 跳出for循环（这里的意思是，不执行 下面的decimal_list.append了）
                continue
            decimal_list.append(integer + '.' + fractor)
        return decimal_list

if __name__ == '__main__':
    sol = Solution()
    res = sol.ambiguousCoordinates3("(00011)")