# Definition of TreeNode:
from enum import Enum


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:

    # Lintcode Easy 481 · Binary Tree Leaf Sum
    def leafSum(self, root):
        """
        由于本质还是在遍历，所以时间复杂度O(n), call stack 空间复杂度 O(h)
        """
        self.res = 0  # 这个属性一般还是在init里面定义较好，这里只是告诉我们，在任意位置定义也可以
        self.helperLeafSum(root)
        return self.res
    def helperLeafSum(self, root):
        if not root: # 比写 if root is None: 更符合 python 编程的规范
            return

            # 如果是叶子，就加值
        if not root.left and not root.right:
            self.res = self.res + root.val

        # 走到这里说明，不是叶子，就可以继续递归调用
        self.helperLeafSum(root.left)
        self.helperLeafSum(root.right)

    # Lintcode Easy 97 · Maximum Depth of Binary Tree 其实就是求二叉树高度
    def maxDepth1(self, root):
        """
        借助递归函数参量
        思路是，求出所有节点的深度，然后取个最大值
        时间复杂度是O(n)， 空间复杂度是 O(h)
        """
        self.maxD = 0
        self.helperMaxDepth(root, 1)
        return self.maxD

    def helperMaxDepth(self, root, height):
        if not root:
            return

        self.maxD = max(self.maxD, height)

        self.helperMaxDepth(root.left, height+1)
        self.helperMaxDepth(root.right, height+1)

    # Lintcode Easy 97 · Maximum Depth of Binary Tree 一句话搞定的递归写法
    def maxDepth2(self, root):
        if root is None:
            return 0

        return max(self.maxDepth2(root.left), self.maxDepth2(root.right)) + 1

    # Lintcode Easy 97 · Maximum Depth of Binary Tree 自己想的思路，不高级但还是写一下
    def maxDepth3(self, root):
        """
        这个解法不是最好的。但这是我一开始想的思路
        就是算到每个 叶子节点的高度，然后取个最大值（不像方法1那样每个节点都取最大值）
        """
        self.maxD = 0
        self.helper(root, 1)
        return self.maxD
    def helper(self, root, height):
        if not root:  # 这句不能省的，因为有的节点，虽然不是叶子节点，但有可能有1个孩子，进到另一个孩子时root是Noone的
            return
        if not root.left and not root.right:
            self.maxD = max(self.maxD, height)
            return
        self.helper(root.left, height+1)
        self.helper(root.right, height+1)

class Weekday(Enum):
    positiveSeq = 1
    negativeSeq = 2

if __name__ == '__main__':
    day1 = Weekday.Mon
    print(day1)

    node1 = TreeNode(1)
    node2 = TreeNode(2)
    node3 = TreeNode(3)
    node4 = TreeNode(4)
    node5 = TreeNode(5)

    node1.left = node2
    node1.right = node3
    node2.left = node4
 #   node2.right = node5


    sol = Solution()

    res = sol.maxDepth1(node1)
    print(res)

    pass