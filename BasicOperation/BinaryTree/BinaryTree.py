# coding=utf-8


class TreeNode:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


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


def traverse_tree(root):
    if root is None:
        return

    print(root.val)
    traverse_tree(root.left)
    traverse_tree(root.right)

# 先序遍历 root left right
def preorder_traverse(root):
    # 8 3 1 6 4 7 10 14 13
    if root is None:
        return

    print(root.val, end=' ')
    preorder_traverse(root.left)
    preorder_traverse(root.right)

'''
空间复杂度
如果在函数里开了个列表，列表里的元素是存在堆中的，就会占用堆空间。
如果函数进行了递归调用，在递归的时候，会占用系统的 调用栈。
所以在分析空间复杂度的时候，要分析 heap + stack 空间加起来。

面试的时候，先问一下面试官，call stack空间算不算程序消耗。
不算的话就是O(1)，没有占用额外的。
算得话，与树的高度呈线性关系，介于logN和N之间（因为N个节点的二叉树画法是多种的）

'''
# 中序遍历 left - root - right
def inorder_traverse(root):
    # 1 3 4 6 7 8 10 13 14
    if root is None:
        return

    inorder_traverse(root.left)
    print(root.val, end=' ')
    inorder_traverse(root.right)

# 后序遍历 left - right - root
def postorder_traverse(root):
    # 1 4 7 6 3 13 14 10 8
    if root is None:
        return

    postorder_traverse(root.left)
    postorder_traverse(root.right)
    print(root.val, end=' ')


# def main():
#     root = build_tree()
#     preorder_traverse(root)
#     print()
#     inorder_traverse(root)
#     print()
#     postorder_traverse(root)
#
#
# if __name__ == '__main__':
#     main()


if __name__ == '__main__':

    n = 4
    result = [0, 1]
    for i in range(n-2):
        b = result[i]+result[i+1]
        result.append(b)

    print(result)
    print( result[-1] )


    root = build_tree()
    preorder_traverse(root)
    print()
    inorder_traverse(root)
    print()
    postorder_traverse(root)