# coding=utf-8


class TreeNode:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

'''
性质：
（1）对BST进行中序遍历 left-root-right 得到得是一个递增(非降)序列
（2）有插入操作（普通的二叉树binary tree没有(因为普通的结构太松散混乱了)，普通的只有深度/宽度优先遍历）
（3）查找高校
（4）常用操作: 插入，查找，删除（用得少）

查找(插入)操作的 时间、空间 复杂度
时间复杂度: 每次操作都会往下走一层，每次走的都是BST的一条路径，深度是多少时间消耗就是多少，是O(h)
空间复杂度：这里在递归的时候才会消耗，没使用栈空间。那递归多少层呢？h层。是O(h)
（二分搜索的思想就是每次扔掉一半儿）

树层数(高度)h与节点数n的关系：
worst case：N（退化为线性）
best case：logN （平衡的BST）

应用是，比如你访问一个元素，你想知道之前有没有访问过，每次访问完就把它插到BST里，
下次访问再看它是不是已经存在BST里，是否contains
所以BST是个很适合于做记录的数据结构，因为插入查找特别快
'''
class BST:

    def __init__(self):
        self.__root = None  # 存根节点代表整棵树，初始状态下是None
        # __两个下划线代表这个变量是private的

    # add在很多地方后，其实也会保持是个BST，那么add在哪里呢？尽量add在叶子上(就不用把树结构拆开重连了),而且,一定可以add在叶子上的！
    def add(self, val):
        self.__root = self.__add_helper(self.__root, val)
        #  这里需要 等号赋值，最开始root是 None，插入后要跟root连接起来，才不会是None

    # 这个 helper 是要返回一个 树(根节点)的
    def __add_helper(self, root, val):
        if not root:                                     # 如果root压根儿就不存在，说明这个树没有
            return TreeNode(val)                         # 那要add的这个val就作为这个树唯一的node返回了
        # 如果传入的root存在，就要开始判断了
        if val < root.val:
            root.left = self.__add_helper(root.left, val)
            # 为什么要连接一下呢？因为有些root.left是None，新插入的节点要返回连接上root.left 这样root.left就不再是None了
            #                   有些root.left不是None，这里 = 连接一下 也还是与原来得节点相连，没啥意义，但是也没副作用。既然有的情况需要连，有的情况不需要连，那就都得连接
        else:
            root.right = self.__add_helper(root.right, val)

        return root

    # 查找   找到返回 ture
    def contains(self, val):
        return self.__contains_helper(self.__root, val)

    def __contains_helper(self, root, val):
        if not root:
            return False
        if root.val == val:
            return True
        elif val < root.val:
            return self.__contains_helper(root.left, val)  # 去左孩子找
            # 这种return就算会一直往回传
        else:
            return self.__contains_helper(root.right, val)   # 去右孩子找

    # leetcode：Insert Node in a Binary Search Tree
    # 这是插入一个 node iterative 的 写法 （还没学。不想看，下次再看）
    def insertNode(self, root, node):
        if root is None:
            return node

        curt = root
        while curt != node:
            if node.val < curt.val:
                if curt.left is None:
                    curt.left = node
                curt = curt.left
            else:
                if curt.right is None:
                    curt.right = node
                curt = curt.right
        return root