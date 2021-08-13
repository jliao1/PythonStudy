# coding=utf-8






from 九章基础班.Tree.BinaryTree二叉树.BinarySearchTree import BST



bst = BST()

bst.add(10)
bst.add(11)

print(bst.contains(10))
print(bst.contains(11))
print(bst.contains(9))