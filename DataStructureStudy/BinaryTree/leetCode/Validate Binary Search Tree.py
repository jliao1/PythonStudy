"""
Definition of TreeNode:
"""
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def __init__(self):
        self.res = []

    # 如果是 valid 的 BST，对它进行中序遍历，得到的如果是一个递增(非递减)序列，就对了
    def isValidBST(self, root):
        # write your code here
        self.inorder_traversal(root)

        # 然后判断 res 是不是严格单调递增
        for i in range(1, len(self.res)):  # 判断范围从1到res的最高位就行. why 1? 因为1处跟前一位比较就行
            if self.res[i] <= self.res[i - 1]:
                return False

        return True

    # 进行一次中序遍历
    def inorder_traversal(self, root):
        if not root:
            return

        self.inorder_traversal(root.left)
        self.res.append(root.val)
        self.inorder_traversal(root.right)



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


if __name__ == '__main__':

    root = build_tree()
    sol = Solution()
    res = sol.isValidBST(root)
    print (res)
