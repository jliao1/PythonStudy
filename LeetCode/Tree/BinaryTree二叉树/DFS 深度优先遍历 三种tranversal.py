class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
# 前中后(在sub tree中视角是)根被遍历的位置
# 子孩子是，默认先左后右哈

# 先序遍历 root left right
def preorder_traverse(root):
    # 8 3 1 6 4 7 10 14 13

    if not root:   # 可译为：如果根本就没有这个root
    # 上面这条语句等价于 if root is None:
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
# 中序遍历 left - root - right  用处是：判断是不是BST对它进行中序遍历，得到的如果是一个递增(非递减)序列，就对了
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

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree 用map
    def findLeaves1(self, root):
        """
        为什么需要用到DFS？
        The order in which the elements (nodes) will be collected in the final
        answer depends on the "height" of these nodes.
        Since height of any node depends on the height of it's children node,
        hence we traverse the tree in a post-order manner (i.e. height of the
        childrens are calculated first before calculating the height of the given node).

        其实leetcode上方法1是，we'll store the pair (height, val) for all the nodes which
        will be sorted later to obtain the final answer. The sorting will be done in
        increasing order considering the height first and then the val. Hence we'll obtain
        all the pairs in the increasing order of their height in the given binary tree.
        但它方法的时间复杂度是 Assuming N is the total number of nodes in the binary tree,
        traversing the tree takes O(N) time. Sorting all the pairs based on their height
        takes O(NlogN) time. Hence overall time complexity of this approach is O(NlogN)

        下面这个方法比较好，跟上面有异曲同工，但去掉了sorting，就 placing each element (val) to
        its correct position in the solution map/array.
        遍历树获取每一个node的depth，再以depth为key储存相同depth的node.val在map里
        时间空间复杂度都是 O(n)
        """
        ans = []
        self.height_map = {}

        maxHeight = self.dfsForFindLeaves3(root)

        for i in range(maxHeight + 1):
            ans.append(self.height_map.get(i))

        return ans
    def dfsForFindLeaves1(self, node):
        # find height
        if node is None:
            return -1

        cur_height = max(self.dfsForFindLeaves3(node.left), self.dfsForFindLeaves3(node.right)) + 1

        if cur_height not in self.height_map:
            # put cur_height into key
            self.height_map[cur_height] = []

        # 对着这个 height key, 添加元素
        self.height_map[cur_height].append(node.val)

        return cur_height

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree 令狐冲版本答案 用 list做的，感觉还是1更好懂
    def findLeaves2(self, root):
        results = []
        self.dfsForFindLeaves2(root, results)
        return results
    def dfsForFindLeaves2(self, root, results):
        if root is None:
            return 0

        level = max(self.dfsForFindLeaves2(root.left, results), self.dfsForFindLeaves2(root.right, results)) + 1
        size = len(results)
        if level > size:
            # 说明要新启一层了
            results.append([])

        # 在 level-1 这个角标的list里 append 进值
        results[level-1].append(root.val)
        return level

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree （炫技解法，有空再看）不需要什么node depth。。python解法硬融两个list。长度不同的list头部对齐，后面用[]补充
    def findLeaves3(self, root):
        if not root:
            return []
        leftList = self.findLeaves3(root.left)
        rightList = self.findLeaves3(root.right)
        newList = []
        # 硬融
        for i in range(max(len(leftList), len(rightList))):
            left = leftList[i] if i < len(leftList) else []
            right = rightList[i] if i < len(rightList) else []
            newList.append(left + right)
        newList.append([root.val])
        return newList


if __name__ == '__main__':

    root = build_tree2()
    sol = Solution()
    l = sol.findLeaves2(root)
    print(l)



