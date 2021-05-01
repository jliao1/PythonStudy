# Leetcode: Validate Binary Search Tree
# 本题思路：如果是 valid 的 BST，对它进行中序遍历，得到的如果是一个递增(非递减)序列，就对了

"""
Definition of TreeNode:
"""
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:

    def __init__(self):
        self.res = []

    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    # 本题思路：如果是 valid 的 BST，对它进行中序遍历，得到的如果是一个递增(非递减)序列，就对了
    def isValidBST(self, root):

        # self.res = []   这个成员变量 res 可以定义在这里，供本class下其他函数直接调用。但写在这里不规范，最好还是写在 init 里

        # 写法一
        self.inorder_traversal(root)
        # # 写法二
        # self.res = self.inorder_traversal(root)

        # 然后判断 res 是不是严格单调递增
        for i in range(1, len(self.res)):  # 判断范围从1到res的最高位就行. why 1? 因为1处跟前一位比较就行
            if self.res[i] <= self.res[i - 1]:
                return False

        return True

    # 进行一次中序遍历
    def inorder_traversal(self, root):
        # 写法1：res是class成员变量的写法
        if not root:
            return
        self.inorder_traversal(root.left)
        self.res.append(root.val)          # 在这里 append 不需要重新 extend在 res 上了，体会下
        self.inorder_traversal(root.right)

        # # 写法2：res是本函数内局部变量的写法
        # res = []
        # if not root:
        #     return res
        #
        # res.extend(self.inorder_traversal(root.left))
        # res.append(root.val)
        # res.extend(self.inorder_traversal(root.right))
        # return res


# 测试用
def build_tree():
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


# 测试用
if __name__ == '__main__':

    #testRoot1 = build_tree()
    sol = Solution()


    testRoot2 = TreeNode(1)
    node2 = TreeNode(1)
    testRoot2.left = node2

    res2 = sol.isValidBST(testRoot2)
    print (res2)
