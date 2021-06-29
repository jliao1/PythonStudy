'''
Given two strings A and B of lowercase letters, return true if and
only if we can swap two letters in A so that the result equals B.
Otherwise, return false.
'''


class Solution:

    # 我自己做的
    def buddyStrings1(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False

        d = 0
        two = []

        items = []
        is_same_tiem = False

        for idx in range(len(s)):
            if s[idx] != goal[idx]:
                d += 1
                two.append(idx)
            if s[idx] in items:
                is_same_tiem = True
            else:
                items.append(s[idx])

        #如果字符串不相同，那么就看是否刚好有两个位置的字符不同，并且这两个位置的字符满足A[i]==B[j] && A[j]==B[i]
        if d == 2 and s[two[0]] == goal[two[1]] and s[two[1]] == goal[two[0]]:
            return True

        # 如果俩字符串相等，那么就找是否有相同的字符
        if d == 0 and is_same_tiem == True:
            return True

        return False

    # 更标准一点的答案，空间复杂度更低
    def buddyStrings2(self, A, B):
        # Write your code here
        if len(A) != len(B):
            return False
        # 如果俩字符串相等，那么就找是否有大雨等于2个以上相同的字符
        if A == B and len(set(A)) < len(A):
            return True
        # 这条语句是找出 A B 中对应不相等的元素
        # 打印出来是元组组成的list [('b', 'c'), ('c', 'b')]
        dif = [(a, b) for a, b in zip(A, B) if a != b]
                       #  取dif[0]的元组  取dif[1]的元组，并将它翻转
        return len(dif) == 2 and dif[0] == dif[1][::-1]

if __name__ == '__main__':
    sol = Solution()
    res = sol.buddyStrings2('aaaaaaabc', 'aaaaaaacb')


