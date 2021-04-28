from BinaryTree import TreeNode

if __name__ == '__main__':
    root = TreeNode.build_tree()
    preorder_traverse(root)
    print()
    inorder_traverse(root)
    print()
    postorder_traverse(root)