# Definition of TreeNode:
from enum import Enum
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# 建立个二叉树，测试用
def build_tree1():
    """
            8
           /  \
          3    10
         / \     \
        1   6     14
            /\    /
           4  7  13

    """
    node_1 = TreeNode(8)
    node_2 = TreeNode(3)
    node_3 = TreeNode(10)
    node_4 = TreeNode(1)
    node_5 = TreeNode(6)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.left = node_2
    node_1.right = node_3

    node_2.left = node_4
    node_2.right = node_5

    node_3.right = node_6

    node_5.left = node_7
    node_5.right = node_8

    node_6.left = node_9

    return node_1
def build_tree2():
    """
        1
       / \
      2   3
         / \
        4   5
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3)
    node_4 = TreeNode(4)
    node_5 = TreeNode(5)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.right = node_2
    node_1.left = node_3
    node_3.left = node_4
    node_3.right = node_5

    return node_1

class Solution:

    # Lintcode Easy 481 · Binary Tree Leaf Sum 有点DFS感觉
    def leafSum1(self, root):
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

    # Lintcode Easy 481 · Binary Tree Leaf Sum 简洁版
    def leafSum2(self, root):

        if not root: return 0

        if not root.left and not root.right:
            return root.val

        return self.leafSum(root.left) + self.leafSum(root.right)

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

    # Lintcode Easy 482 · Binary Tree Level Sum 自己用BFS做的
    def levelSum1(self, root, level):
        """
        用 BFS 做的，时间复杂度固定是O(n), 空间复杂度是 O(n)
        我这方法是遍历所有的，存所有层的，好处是想看哪层，改变一下input的level就行
        方法2是用DFS做的，时间空间复杂度和 level 相关(level更大的层的nodes不需要遍历了)
        """
        if not root:
            return 0

        all_level_lists = []

        q = deque()

        q.append(root)

        while len(q) != 0:

            cur_level_list = []
            n = len(q)

            for _ in range(n):
                cur_node = q.popleft()
                cur_level_list.append(cur_node.val)

                if cur_node.left:
                    q.append(cur_node.left)
                if cur_node.right:
                    q.append(cur_node.right)

            if cur_level_list:
                all_level_lists.append(cur_level_list)

        # 这个很容易忘记
        return sum(all_level_lists[level - 1]) if level <= len(all_level_lists) and level >= 1 else 0

    # Lintcode Easy 482 · Binary Tree Level Sum 用DFS做的
    def levelSum2(self, root, level):
        """
        这是用DFS和divide and conquer做的
        为什么可以用DFS做？ 好像看 左右 child nodes 的，都可以用DFS
        """
        if root is None:
            return 0
        elif level == 1:
            return root.val
        return self.levelSum2(root.left, level-1) + self.levelSum2(root.right, level-1)


if __name__ == '__main__':
    root = build_tree1()
    sol = Solution()
    l = sol.levelSum2(root, 3)
    print(l)

    pass