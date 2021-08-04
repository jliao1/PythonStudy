# Lettcode：Bash Game
# 思路 Bash博弈先手必败的条件就是n%(m+1)==0,我们来分析一下这个情况。首先假设n=m+1，
#     这个时候无论先手第一次拿多少个，由于剩下的物品数量一定是小于等于m的，所以后手一
#     定可以一次拿完。所以这个时候先手必败。如果n是m+1的倍数，其实也很简单的，因为无论
#     先手一次拿多少个物品，后手总是可以做出这样的策略：拿的物品和先手加在一起恰好是m+1
#     个，这样若干回合之后哦一定又会回到我们说的第一种情况，先手必败。所以结论得证。
# 网址分析参考：https://pipilove.gitbooks.io/leetcode/content/ba-shi-bo-595528-bash-game.html


# reccursive版本：没懂
def canWinBash( n):
    if n <= 0:
        return False
    if n <= 3:
        return True
            # 如果总量有 n-1 我输了    如果总量有 n-2 我输了    如果总量有 n-3 我输了
    return not canWinBash(n-1) or not canWinBash(n-2) or not canWinBash(n-3)


# def canWinBash(self, n):
#     # Write your code here
#     if n % 4 == 0:
#         return False
#     else:
#         return True


if __name__ == '__main__':
    str = " aa   bbbbb         ccc  d"
    str_list1 = str.split()
    print(str_list1)  # 打印出来是 ['aa', 'bbbbb', 'ccc', 'd']
    str_list2 = str.split(' ')
    print(str_list2)  # 打印出来是 ['', 'aa', '', '', 'bbbbb', '', '', '', '', '', '', '', '', 'ccc', '', 'd']
