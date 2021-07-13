import copy

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
      3   2
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
def build_tree3():
    """
        1
       / \
    null  null
    """
    node_1 = TreeNode(1)

    return node_1

class Solution:

    # lintcode(力扣94) Easy 67 · Binary Tree Inorder Traversal  这道题挑战让你用iterative写法
    def inorderTraversal_iterative(self, root):
        """
        时间空间都是 O(n)
        """

        if root is None:
            return []

        # 创建一个 dummy node，右指针指向 root
        # 并放到 stack 里，此时 stack 的栈顶 dummy
        # 是 iterator 的当前位置
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]

        inorder = []
        # 每次将 iterator 挪到下一个点
        # 也就是调整 stack 使得栈顶到下一个点
        while stack:
            node = stack.pop()
            if node.right:
                node = node.right
                # 把左孩子都压栈
                while node:
                    stack.append(node)
                    node = node.left
            if stack:
                inorder.append(stack[-1].val)

        return inorder

    # lintcode(力扣144) Easy 66 · Binary Tree Preorder Traversal  这道题挑战让你用iterative写法
    def preorderTraversal_iterative(self, root):
        """
        时间O(n)
        空间O(n)
        depending on the tree structure, we could keep up to the entire tree, therefore, the space complexity is O(n)
        """
        if root is None:
            return []
        stack = [root]
        preorder = []
        while stack:
            node = stack.pop()
            preorder.append(node.val)

            # 因为 stack 是先进后出。要先访问 left 后访问 right，所以 right 应该先进 stack
            if node.right:
                stack.append(node.right)

            # 然后 left 再进 stack (这样才能在 pop 的时候，才能 pop)
            if node.left:
                stack.append(node.left)
        return preorder

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree 用map
    def findLeaves1(self, root):
        """
        为什么需要用到DFS？
        因为貌似需要先访问孩子，再访问根节点。有点post order DFS 的味道

        The order in which the elements (nodes) will be collected in the final
        answer depends on the "height" of these nodes.
        Since height of any node depends on the height of it's children node,
        hence we traverse the tree in a post-order DFS manner (i.e. height of the
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

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree （炫技解法，有空再看）
    def findLeaves3(self, root):
        """
        不需要什么node depth。。python解法硬融两个list。长度不同的list头部对齐，后面用[]补充
        """
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

    # lintcode Easy 376 · Binary Tree Path Sum (与力扣112 path sum类似)
    def binaryTreePathSum1(self, root: TreeNode, targetSum: int) -> bool:
        """
        时间复杂度：O(n)，其中 n是节点的数量。我们每个节点只访问一次，因此时间复杂度为 O(n)。
        空间复杂度：取决于最终返回结果的res的空间大小。
                  最差情况下，当树为fully complete二叉树且每条路径都符合要求时，
                  空间复杂度为O(Nlog(N))。 高度为logN，路径条数为 N/2 (最后一层的叶子节点数)
        """
        self.t = targetSum
        self.lists = []

        self.dfs_for_binaryTreePathSum(root, [], 0)

        return self.lists
    def dfs_for_binaryTreePathSum1(self, root, pre_list, pre_sum):
        if root:
            cur_sum = pre_sum + root.val
        else:
            return

        cur_list = pre_list + [root.val]

        if cur_sum == self.t and not root.left and not root.right:
            self.lists.append(cur_list)
            return
        else:
            self.dfs_for_binaryTreePathSum(root.left, cur_list, cur_sum)

            self.dfs_for_binaryTreePathSum(root.right, cur_list, cur_sum)

    # lintcode Easy 376 · Binary Tree Path Sum  删path的写法
    def binaryTreePathSum2(self, root, target):
        if not root:
            return []
        res = []
        self.dfs_for_binaryTreePathSum2(root, target, [], res)
        return res
    def dfs_for_binaryTreePathSum2(self, root, target, path, res):
        # 空节点
        if root is None:
            return
        path.append(root.val)
        # 叶节点
        if not root.left and not root.right:
            if target == root.val:
                res.append(path[:])  # 新的内存
            del path[-1]  # 原内存
            return
        # 非叶节点
        self.dfs_for_binaryTreePathSum2(root.left, target - root.val, path, res)
        self.dfs_for_binaryTreePathSum2(root.right, target - root.val, path, res)
        del path[-1]

    # 力扣112 Easy path sum
    def hasPathSum(self, root, sum):
        """
        Time complexity : we visit each node exactly once, thus the time complexity is O(N),
                          where N is the number of nodes.
        Space complexity : In the worst case, the tree is completely unbalanced,
                            e.g. each node has only one child node, the recursion call would occur N times
                            (the height of the tree), therefore the storage to keep the call stack would be O(N).
                            But in the best case (the tree is completely balanced), the height of the tree would be log(N).
                            Therefore, the space complexity in this case would be O(log(N)).
        """
        if not root:
            return False

        sum -= root.val

        # If node is a leaf, one checks if the the current sum is zero
        if not root.left and not root.right:
            return sum == 0

        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    # lintcode(力扣257) Easy 480 · Binary Tree Paths 自己的解法
    def binaryTreePaths1(self, root: TreeNode):
        """
        这是我自己写的，好处是，传参数时，
        不会把越来越长的 curr_path 传进去
        而且遍历完一个node后，curr_path会删减(回溯)

        时间复杂度是O(n)因为DFS会访问每个节点至少1次

        空间复杂度，call stack(函数调用栈)取决于递归的次数，最好情况下logN，最差情况下O(n)因为是一个linked list了
        然后还有path_lists contains as many elements as leafs in the tree and hence couldn't be larger than
        logN for the trees containing more than one element. Hence the space complexity is determined
        by a stack call. （这句话没看懂）
        """
        self.path_lists = []
        self.curr_path = []

        self.dfsForBinaryTreePaths1(root)

        return self.path_lists
    def dfsForBinaryTreePaths1(self, root: TreeNode):
        if not root:
            return

        self.curr_path.append(str(root.val))

        # 如果是 叶子节点, 需要处理一下 append 进 path_lists 里
        #      注意这一坨代码如果要放在这个位置，必须要 curr_path.pop() 一下，不然 curr_path 不能得到及时的更新
        if not root.left and not root.right:
            self.path_lists.append('->'.join(self.curr_path))
            self.curr_path.pop()
            return

        self.dfsForBinaryTreePaths1(root.left)
        self.dfsForBinaryTreePaths1(root.right)

        # 如果对叶子节点的处理放在这个位置，就不用 curr_path.pop()，因为反正下面已pop了

        # 处理完这个 root node 后，再把它从 curr_path 删掉
        self.curr_path.pop()

    # lintcode(力扣257) Easy 480 · Binary Tree Paths 令狐和leetcode上的答案差不多是这样
    def binaryTreePaths2(self, root: TreeNode):
        """
        这个解法，写得会简单点，但传入construct_path()的参数 path 会越来越大，
        不过，面试的时候，这样写好像也无所谓.
        """
        self.path_lists = []

        def construct_path(root, curr_path):

            if not root:
                return

            curr_path.append(str(root.val))

            if not root.left and not root.right:
                self.path_lists.append('->'.join(curr_path))
            else:                         # 这个一定要写deep copy才可以pass by value
                construct_path(root.left, copy.deepcopy(curr_path))
                construct_path(root.right, copy.deepcopy(curr_path))

        construct_path(root, [])

        return self.path_lists

    # lintcode(力扣100) Easy 469 · Same Tree自己写的答案一次过
    def isIdentical(self, a, b):
        """
        Time complexity : O(N), where N is a number of nodes in the tree, since one visits each node exactly once.
        Space complexity : O(log(N)) in the best case of completely balanced tree, O(N) in the worst case of completely unbalanced tree, to keep a recursion stack.
        """
        if not a and not b:
            return True

        if a and b and self.isSameVal(a, b):
            return self.isIdentical(a.left, b.left) and self.isIdentical(a.right, b.right)
        else:
            return False
    def isSameVal(self, node1, node2):
        if node1.val == node2.val:
            return True
        else:
            return False

if __name__ == '__main__':

    root = build_tree2()
    l1 = inorder_traverse(root)
    print(l1)

    sol = Solution()
    l2 = sol.inorderTraversal_iterative(root)
    print(l2)



    pass



