import copy
import collections
from collections import deque
# 建立个二叉树，测试用
def build_tree1():
    """
            8
           /  \
          3    10
         / \     \
        1   6     14
        \   /\    /
        11 4  7  13
          \
          12
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
    node_10 = TreeNode(11)
    node_11 = TreeNode(12)

    node_1.left = node_2
    node_1.right = node_3

    node_2.left = node_4
    node_2.right = node_5

    node_3.right = node_6

    node_5.left = node_7
    node_5.right = node_8

    node_6.left = node_9
    node_4.right = node_10
    node_10.right = node_11

    return node_1
def build_tree2():
    """
        4 (3)
       / \
  （2)2   5 (0)
     / \
 (0)1   3 (1)
         \
         14 (0)
    """
    node_1 = TreeNode(4)
    node_2 = TreeNode(2)
    node_3 = TreeNode(5)
    node_4 = TreeNode(1)
    node_5 = TreeNode(3)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.left = node_2
    node_1.right = node_3
    node_2.left = node_4
    node_2.right = node_5
    node_5.right = node_6

    return node_1
def build_tree3():
    """
        1
       / \
      2   3
     /
    4
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3)
    node_4 = TreeNode(4)

    node_1.left = node_2
    node_1.right = node_3
    node_2.left = node_4

    return node_1
def build_tree4():
    """
        1
       / \
      3   2
           \
            4
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(3)
    node_3 = TreeNode(2)
    node_4 = TreeNode(4)

    node_1.left = node_2
    node_1.right = node_3
    node_3.right = node_4

    return node_1

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 知识点
'''
- 前中后(在sub tree中视角是)遍历，是根被遍历的位置, 子树默认是先左后右哈
- 时间复杂度O(n)
- 空间复杂度是 O(h), call stack最深存树的高度个 recursion function
'''

'''
空间复杂度
如果在函数里开了个列表，列表里的元素是存在堆中的，就会占用堆空间。
如果函数进行了递归调用，在递归的时候，会占用系统的 调用栈。
所以在分析空间复杂度的时候，要分析 heap + stack 空间加起来。

面试的时候，先问一下面试官，call stack空间算不算程序消耗。
不算的话就是O(1)，没有占用额外的。
算得话，与树的高度呈线性关系，介于logN和N之间（因为N个节点的二叉树画法是多种的）
'''

# 先序遍历 root left right
def preorder_traverse(root):
    # 8 3 1 6 4 7 10 14 13

    if not root:   # 可译为：如果根本就没有这个root
    # 上面这条语句等价于 if root is None:
        return

    print(root.val, end=' ')
    preorder_traverse(root.left)
    preorder_traverse(root.right)

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

# lintcode(力扣173) Medium 86 · Binary Search Tree Iterator按中序遍历的方式做Inorder traversal，其实这就是中序遍历iterative的写法
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        # 初始化时从root开始一路向左走到最末端的节点，即全局最小值，并将一路访问过的节点加入到stack中
        self._to_most_left_min(root)

    def hasNext(self):
        # stack[-1] 一直存放 iterator 指向的当前节点。因此在判断有没有下一个节点时，只需要判断 stack 是否为空。
        return len(self.stack) > 0   # 或者写: return bool(self.stack)

    # worst case O(h)，均摊下来访问每个节点的时间复杂度时O(1)
    def _next(self):
        # 异常检测
        if len(self.stack) == 0: return None

        next_node = self.stack.pop()

        if next_node.right:  # 这句话不加其实也行，因为 next_node.right 是None的话也可以处理的
            self._to_most_left_min(next_node.right)
        return next_node

    # 找全局最小值，一路向左，并将一路访问过的节点加入到stack中
    # 这样最后一个加进去的当然就是最小值
    def _to_most_left_min(self, root):
        while root:
            self.stack.append(root)
            root = root.left

class Solution:
    def __init__(self):
        self.counter = 0
        self.string = []   # lintcode(力扣606) Easy 1137 · Construct String from Binary Tree

    # lintcode(力扣94) Easy 67 · Binary Tree Inorder Traversal  这道题挑战让你用iterative写法
    def inorderTraversal_iterative1(self, root):
        stack = []
        res = []

        while stack or root:  # 直到栈空和root是None就停止
            # 先往左走，边走边入栈
            while root:
                stack.append(root)
                root = root.left
            # 弹栈
            root = stack.pop()
            # 遍历
            res.append(root.val)
            # 后root往右子树走一步
            root = root.right

        return res

    # lintcode(力扣94) Easy 67 · Binary Tree Inorder Traversal  这道题挑战让你用iterative写法
    def inorderTraversal_iterative2(self, root):
        if not root:
            return []

        def to_leftmost(root):
            while root:
                stack.append(root)
                root = root.left

        res = []
        stack = []
        to_leftmost(root)

        while stack:
            node = stack.pop()
            res.append(node.val)
            to_leftmost(node.right)

        return res

    # lintcode Easy 66 · Binary Tree Preorder Traversal  这道题挑战让你用iterative写法
    def preorderTraversal_iterative(self, root):
        """
        时间O(n)
        空间O(h)  depending on the tree structure, we could keep up to the entire tree, therefore, the space complexity is O(n)
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

    # lintcode(力扣145) Easy 68 · Binary Tree Postorder Traversal
    def postorderTraversal_iterative1(self, root):
        """
        后续遍历是先左子树，再右子树再到父结点，倒过来看就是先父结点，再右子树再左子树。
        是前序遍历改变左右子树访问顺序。 再将输出的结果倒序输出一遍就是后序遍历。
        如果不只是求后序遍历，而是要后序遍历的过程中每个点做额外的操作，那么这种办法就不好使了, 看下方法2更genral
        """
        if root is None:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        return result[::-1]

    # lintcode(力扣145) Easy 68 · Binary Tree Postorder Traversal 这个好难…
    def postorderTraversal_iterative2(self, root):
        """这个相比之下更好理解吧"""
        if not root:
            return []

        def to_left_most_and_to_right_most(curNode):
            while curNode:
                stack.append(curNode)
                # 能左走就左走，走不了(None)的话就向右走，直到触底
                curNode = curNode.left if curNode.left else curNode.right

        ans, stack, cur = [], [], root

        to_left_most_and_to_right_most(cur)

        while stack:
            cur = stack.pop()
            ans.append(cur.val)
            # 如果 cur 栈顶节点stack[-1] 的左孩子 （如果不是就继续while循环弹栈）
            if stack and stack[-1].left == cur:
                # 栈顶节点stack[-1]，先hold住不出栈
                # cur往 栈顶(父)节点 的右孩子走一步
                cur = stack[-1].right
                to_left_most_and_to_right_most(cur)

        return ans

    # lintcode(力扣145) Easy 68 · Binary Tree Postorder Traversal
    def postorderTraversal_iterative3(self, root):
        """这思路好难懂"""
        result = []
        stack = []

        curNode = root
        while stack or curNode:
            # 能左就左。继续左没有时，就向右一步
            while curNode:
                stack.append(curNode)

                if curNode.left:
                    curNode = curNode.left
                else:
                    curNode = curNode.right

            # pop stack，添加到结果
            curNode = stack.pop()
            result.append(curNode.val)

            # 栈不空, 且curNode是栈顶stack[-1]的左子节点，
            #               转到栈顶stack[-1]的右兄弟，否则退栈
            if stack and stack[-1].left == curNode:
                curNode = stack[-1].right
            else:
                curNode = None

        return result

    # lintcode（力扣105）Medium 73 · Construct Binary Tree from Preorder and Inorder Traversal 这种题处理 index要疯，唉
    def constructFromPreInorder1(self, preorder, inorder):
        """
        九章的答案
        前序的第一个为根，在中序中找到根的位置。
        中序中根的左右两边即为左右子树的中序遍历。同时可知左子树的大小size-left。
        前序中根接下来的size-left个是左子树的前序遍历。
        由此可以递归处理左右子树。
        """
        if not inorder: return None # inorder is empty
        root = TreeNode(preorder[0])
        root_index = inorder.index(preorder[0])
        root.left = self.constructFromPreInorder1(preorder[1: 1 + root_index], inorder[: root_index])
        root.right = self.constructFromPreInorder1(preorder[root_index + 1:], inorder[root_index + 1:])
        return root

    # lintcode（力扣105）Medium 73 · Construct Binary Tree from Preorder and Inorder Traversal
    def constructFromPreInorder2(self, preorder, inorder):
        """
        * 我写了个in-place版本的，思路跟官方答案一样，但运行起来比官方答案快，可读性也不错。
        * 官方答案思路清晰，但总是这样preorder[1: 1 + root_index] 还有这样inorder[: root_index]对 list 进行切片，这操作会每次copy部分原list产生一次新的list，这会很增加rum time的。
        * 所以我写了一个in-place版本的，基于preorder 和 inorder这两个原 list 操作就行，递归函数传参时不会产生新list
        * 由于不需要list切片，只需要visit each node once，所以时间复杂度是O(N)
        * 空间复杂度是O(h)，h是树高度在logN～N之间，这是call stack使用的空间
        """
        # 如果输入为空 或 2个列表 不一样长，return None
        if not inorder and len(inorder) != len(preorder):
            return None

        self.preorder_index_tracking = 0

        # 这个分治函数参数 left_index 和 right_index 看的是 inorder list 的 index
        def divide_conquer(left_index, right_index):
            if left_index > right_index: return None

            # build root node
            root_value = preorder[self.preorder_index_tracking]
            root = TreeNode(root_value)

            # process index
            root_index_in_inorder = inorder.index(preorder[self.preorder_index_tracking])
            self.preorder_index_tracking += 1

            # 分治开始啦, 这里主要是看 inorder 序列
            root.left = divide_conquer(left_index, root_index_in_inorder - 1)
            root.right = divide_conquer(root_index_in_inorder + 1, right_index)

            return root

        return divide_conquer(0, len(inorder) - 1)

    # lintcode（力扣106）Medium 72 Construct Binary Tree from Inorder and Postorder Traversal
    def constructFromPstInorder1(self, inorder, postorder):
        if not inorder: return None # inorder is empty
        root = TreeNode(postorder[-1])
        rootPos = inorder.index(postorder[-1])
        root.left = self.buildTree(inorder[ : rootPos], postorder[ : rootPos])
        root.right = self.buildTree(inorder[rootPos + 1 : ], postorder[rootPos : -1])
        return root

    # lintcode（力扣106）Medium 72 Construct Binary Tree from Inorder and Postorder Traversal
    def constructFromPstInorder2(self, inorder, postorder):
        if not inorder and len(inorder) != len(postorder):
            return None

        def build(in_start, in_end, post_start, post_end):
            if in_start > in_end or post_start > post_end:
                return None

            root_value = postorder[post_end]
            root = TreeNode(root_value)

            # 下面这句话好像不写也可以，但写上也挺有助于理解
            if in_start == in_end or post_start == post_end:
                return root

            root_index_in_inorder = inorder.index(root_value)
            # 左子树的长度，用来定位index范围的
            left_len = root_index_in_inorder - in_start

            #                左子树在inorder列表里的index范围          左子树在postorder列表里的index范围
            root.left = build(in_start, root_index_in_inorder - 1, post_start, post_start + left_len - 1)
            #                右子树在inorder列表里的index范围         右子树在postorder列表里的index范围
            root.right = build(root_index_in_inorder + 1, in_end, post_start + left_len, post_end - 1)

            return root

        return build(0, len(inorder) - 1, 0, len(postorder) - 1)

    # lintcode（力扣889）Medium 1593 Construct Binary Tree from Preorder and Postorder Traversal
    def constructFromPrePost1(self, pre, post):
        """
        左边分支有L个节点。左边分支头节点是pre[1],但是也是左边分支后序遍历的最后一个。
        所以pre[1] = post[L-1]。因此，L = post.indexOf(pre[1]) + 1。
        在递归过程中，左边分支节点位于pre[1 : L + 1]和post[0 : L]中，
        右边分支节点位于pre[L+1 : N]和post[L : N-1]中。(不包括区间右端点)
        但这个不是in-place的，需要copy
        """
        if not pre: return None
        root = TreeNode(pre[0])
        if len(pre) == 1: return root

        L = post.index(pre[1]) + 1
        root.left = self.constructFromPrePost1(pre[1:L + 1], post[:L])
        root.right = self.constructFromPrePost1(pre[L + 1:], post[L:-1])
        return root

    # lintcode（力扣889）Medium 1593 Construct Binary Tree from Preorder and Postorder Traversal
    def constructFromPrePost2(self, pre, post):
        """
        in-place写法，时间复杂度O(n),空间复杂度O(h)
        """
        if not pre and len(pre) != len(post):
            return None

        def build_tree(pre_start, pre_end, post_start, post_end):
            if pre_start > pre_end or post_start > post_end:
                return None

            # bulid root
            root = TreeNode(pre[pre_start])

            # 这步不能省略，在处理最后2个节点的时候，如果没了这不，就过头了，错乱了
            if pre_start == pre_end or post_start == post_end:
                return root

            # process index
            left_root_Index_In_Pre = pre_start + 1
            left_root_value = pre[left_root_Index_In_Pre]
            left_root_Index_In_post = post.index(left_root_value)
            #                      left_len_in_pre = left_len_in_post = left_root_Index_In_post - post_start + 1
            left_end_Index_In_Pre = left_root_Index_In_Pre + (left_root_Index_In_post - post_start) # 这里是算 index 比 长度少1

            # 前序的分法，就是把第一个 root value 拿去建 root后剔除它，然后剩下的，左部分是 left sub tree，右是 right sub tree
            # 后序的分法，就是把最后一位是 root value 剔除，剩下的 左部分是 left sub tree，右是 right sub tree

            #                     the range of left sub tree in preorder list  为什么left_root_Index_In_Pre是pre_start+1？因为前序的分法，就是把第一个 root value 拿去建 root后剔除它
            root.left = build_tree(left_root_Index_In_Pre,               left_end_Index_In_Pre, \
            #                      the range of left sub tree in inorder list
                                   post_start,                  left_root_Index_In_post )

            #                      the range of right sub tree in preorder list
            root.right = build_tree(left_end_Index_In_Pre+1,     pre_end, \
            #                       the range of right sub tree in inorder list
                                    left_root_Index_In_post + 1, post_end - 1)  # 为什么要 post_end-1? 因为后序的分法，就是把最后一位是 root value 剔除
            return root

        return build_tree(0, len(pre) - 1, 0, len(post) - 1)

    # lintcode(力扣366) Medium 650 · Find Leaves of Binary Tree 自顶向上一层一层返回leaves,用map做,每一层key是高度，values是所有叶子集合
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

              4 (3)
             / \
        （2)2   5 (0)
           / \
        (0)1   3 (1)
                \
                14 (0)

        """
        ans = []
        self.height_map = {}

        maxHeight = self.dfsForFindLeaves1(root)

        for i in range(maxHeight + 1):
            ans.append(self.height_map.get(i))

        return ans
    def dfsForFindLeaves1(self, node):
        # find height
        if node is None:
            return -1

        cur_height = max(self.dfsForFindLeaves1(node.left), self.dfsForFindLeaves1(node.right)) + 1

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

    # 力扣112 Easy · path sum
    def hasPathSum(self, root, sum):
        """
        Time complexity : we visit each node exactly once, thus the time complexity is O(N),
                          where N is the number of nodes.
        Space complexity : 取决于高度啦
                            In the worst case, the tree is completely unbalanced,
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
    def isIdentical(self, T1, T2):
        """
        Time complexity : O(N), where N is a number of nodes in the tree, since one visits each node exactly once.
        Space complexity : O(log(N)) in the best case of completely balanced tree, O(N) in the worst case of completely unbalanced tree, to keep a recursion stack.
        """
        if not T1 and not T2:
            return True

        if T1 and T2 and T1.val == T2.val:
            return self.isIdentical(T1.left, T2.left) and self.isIdentical(T1.right, T2.right)

        return False

    # lintcode(力扣606) Easy 1137 · Construct String from Binary Tree
    def tree2str(self, t):
        """
        类似于树的前序遍历，只需在遍历时在左子树和右子树最外面加一对括号即可。
        注意如果右子树为空，则右子树不需要加括号；若左子树为空而右子树非空，则需要在右子树前加一对空括号表示左子树。
        """
        if not t:
            return ''

        self.string.append(str(t.val))

        if t.left:
            self.string.append('(')
            self.tree2str(t.left)
            self.string.append(')')

        if t.right:
            if not t.left:
                self.string.append('()')

            self.string.append('(')
            self.tree2str(t.right)
            self.string.append(')')

    # lintcode(力扣606) 470 · Tweaked Identical Binary Tree
    def isTweakedIdentical1(self, a, b):
        """
        这是我自己的思路，进行中序遍历（value更小的，或者有value的，先遍历）
        最后再比对两个树的结果，是否一样
        因为是dfs所以时间复杂度O(n)，这个思路空间复杂度也是O(n)啦
        可以看看那版本2令狐冲老师的思路
        """
        self.res1 = []
        self.res2 = []

        self.preorderDFS(a, self.res1)
        self.preorderDFS(b, self.res2)

        return self.res1 == self.res2
    def preorderDFS(self, root, res):
        self.counter += 1
        if not root:
            res.append(None)
            return

        res.append(root.val)

        if root.left and root.right:
            # 如果左右子树都存在，先遍历 value 更小的
            if root.left.val <= root.right.val:
                self.preorderDFS(root.left, res)
                self.preorderDFS(root.right, res)
            else:
                # 再遍历 value 更大的
                self.preorderDFS(root.right, res)
                self.preorderDFS(root.left, res)
        elif root.right and not root.left:
            # 如果右子树存在，左子树不存在，先右再左
            self.preorderDFS(root.right, res)
            self.preorderDFS(root.left, res)
        elif root.left and not root.right:
            # 如果左子树存在，右子树不存在，先遍历 左 再右
            self.preorderDFS(root.left, res)
            self.preorderDFS(root.right, res)
        elif not root.left and not root.right:
            # 如果左右子树都不存在，先左再右
            self.preorderDFS(root.left, res)
            self.preorderDFS(root.right, res)

    # lintcode(力扣606) 470 · Tweaked Identical Binary Tree
    def isTweakedIdentical2(self, a, b):
        """
        判断该子树不交换左右子树和交换左右子树是否能与对应的子树一致，往下一个一个判断即可
        """
        self.counter += 1
        # 情况1：a和b都空
        if not a and not b:
            return True
        # 情况2：a和b都不空
        if a and b and a.val == b.val:
            # 不交换左右子树
            r1 = self.isTweakedIdentical2(a.left, b.left)
            r2 = self.isTweakedIdentical2(a.right, b.right)
            # 交换了左右子树
            r3 = self.isTweakedIdentical2(a.left, b.right)
            r4 = self.isTweakedIdentical2(a.right, b.left)
            return r1 and r2 or r3 and r4
        # 情况3：剩下的情况直接return False
        return False

    # Lintcode(力扣663) Medium 864 · Equal Tree Partition 这是leetcode的解法，有点巧妙，用list来存数据的
    def checkEqualTree1(self, root: TreeNode) -> bool:
        """
        用一个list存每个subtree的sum，最后pop掉root的sum。
        如果list裡有root sum / 2 return True else False。
        时间O(n)
        空间是O(n), the size of seen = []
        """
        seen = []

        def sum_(node):
            if not node:
                return 0
            # postorder_traverse
            seen.append(sum_(node.left) + sum_(node.right) + node.val)
            return seen[-1]

        total = sum_(root)
        seen.pop()  # 把最后一个去掉，因为它是整个树的sum
        return total / 2.0 in seen

    # Lintcode(力扣663) Medium 864 · Equal Tree Partition 我自己写的答案
    def checkEqualTree2(self, root: TreeNode) -> bool:
        """
        思路是：
        我们可以先从根节点进行一次遍历, 然后就可以得到整棵树的节点值总和 sum
        然后再进行一次遍历, 在这一次遍历的过程中判断每个节点的子树节点值总和是否等于sum/2, 如果有返回true即可
        """
        if not root:
            return 0

        self.whole_tree_sum = self.sum1(root)

        self.isEqual = False
        self.sum2(root.left)
        self.sum2(root.right)

        return self.isEqual
    def sum2(self, root):
        """
        这个部分有点难写好
        """
        if not root:
            return 0

        '''
        这个方法只有判断左右孩子不为空，才进入sum2()。不然进入了后return的是0，有可能整个树的sum也是0的
        '''
        if not root.left and not root.right:
            sum = root.val

        if root.left and not root.right:
            sum = root.val + self.sum2(root.left)

        if root.right and not root.left:
            sum = root.val + self.sum2(root.right)

        if root.left and root.right:
            sum = root.val + self.sum2(root.left) + self.sum2(root.right)

        if sum * 2 == self.whole_tree_sum:
            self.isEqual = True

        return sum
    def sum1(self, root):
        if not root:
            return 0

        return root.val + self.sum1(root.left) + self.sum1(root.right)

    # Lintcode(力扣663) Medium 864 · Equal Tree Partition 九章答案, 用map做的
    def checkEqualTree3(self, root):
        """
        借助 set (或 map) 只进行一次遍历, 即第一次遍历时把所有的权值总和放到 set 里,
        最后获得整棵树的节点值总和时, 直接判断集合里是否有 sum / 2 即可.
        """
        self.mp = {}
        sum_of_tree = self.dfsForCheckEqualTree3(root)

        # edge case 1: sum是0的情况要特殊处理一下（因为0处以2依然等于2）
        if sum_of_tree == 0:
            # 如果 sum_of_tree 是 0，只要有子树sum是0的个数大于1个就满足
            return self.mp[0] > 1

        # normal case：
        return (sum_of_tree/2) in self.mp
    def dfsForCheckEqualTree3(self, root):
        if not root:
            return 0
        curr_sum = root.val + self.dfsForCheckEqualTree3(root.left) + self.dfsForCheckEqualTree3(root.right)

        if curr_sum in self.mp:
            self.mp[curr_sum] += 1
        else:
            self.mp[curr_sum] = 1

        return curr_sum

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

    # Lintcode Medium 94 · Binary Tree Maximum Path Sum
    def maxPathSum(self, root):
        """
        平时遇到树结构90%都可以用分治做出来
        一个subtree是一个小组
        可以想想能不能先搞定左孩子，再搞定右孩子，再搞定父节点（post DFS赶脚）

        在遍历树的过程中，要看两样，
        一是当前subtree子树的最大值，设为全局变量更好
        一是当前subtree能对parent tree父树贡献什么，挑选左右子树谁更大，是分治法

        由于是dfs所以时间复杂度是O(n)啦
        与领扣1181很像
        """
        if not root:
            return 0

        self.max_sum = -float('inf')

        self.postorder_dfs(root)

        return self.max_sum
    def postorder_dfs(self, root):
        if not root: return 0
        left = self.postorder_dfs(root.left)
        right = self.postorder_dfs(root.right)

        # 返回上一层级的值，只能是：    自己      自己带右孩子        自己带左孩子
        curr_max_to_parent = max(root.val, root.val + right, root.val + left)  # 由于带孩子可能使得这个小tree的值减小，所以有可能不带孩子

        # 返回前 更新一下max_sum
        self.max_sum = max(self.max_sum, root.val + left + right, curr_max_to_parent)

        return curr_max_to_parent

    # 自练 领扣 E 1181 · Diameter of Binary Tree
    def diameterOfBinaryTree(self, root):
        """
        时间,空间复杂度O(n)
        与领扣94很，做的次数：1，8.29做的
        """
        self.length = 0
        if not root:
            return self.length

        self.height(root)

        return self.length
    def height(self, root):
        if not root:
            return 0

        left_height = self.height(root.left)
        right_height = self.height(root.right)

        self.length = max((left_height + right_height), self.length)

        return max(left_height, right_height) + 1


    # Lintcode Medium(力扣426) 1534 · Convert Binary Search Tree to Sorted Doubly Linked List
    def treeToDoublyList(self, root):
        """
        Each node in a doubly linked list has a predecessor and successor

        The left pointer 指向前 of the tree node should point to its predecessor,
        and the right pointer 指向后 should point to its successor

        这是我自己左的，感觉挺好
        """
        if not root:
            return root

        head, tail = self.dfs_treeToDoublyList(root)

        # 连接首尾
        tail.right = head
        head.left = tail

        return head
    def dfs_treeToDoublyList(self, root):
        """
        是中序遍历，
        思路：
        由于进到右子树，需要返回右子树变成双链表的头 给上一层级用
            进到左子树，需要返回左子树变成双链表的尾 给上一层级用
        然后又要一直记录 双链表的最前端返回给主函数用
        而且在主函数内也要连接整个双链表的 最头 最尾
        所以这个函数就 返回 头 尾 一直记录着好啦
        """
        if not root:
            return root, root

        # 它俩要先赋值，因为如果 下面2个if语句都没进入执行，它俩得有个初始值
        left_head, right_tail = root, root

        # 如果右孩子存在
        if root.right:
            # dfs 返回的 后续会用到 sub_right_head 接回 root
            right_head, right_tail = self.dfs_treeToDoublyList(root.right)
            # 连上（处理root右边情况）
            right_head.left = root
            root.right = right_head   # 这句话好像多余

        # 如果左孩子存在
        if root.left:
            # dfs 返回的 后续会用到 sub_left_tail 接回 root
            left_head, left_tail = self.dfs_treeToDoublyList(root.left)
            # 连上（处理root左边情况）
            left_tail.right = root
            root.left = left_tail

        return left_head, right_tail

    # Lintcode(力扣652) Medium 1108 · Find Duplicate Subtrees 给两个树,return all duplicate subtrees
    def findDuplicateSubtrees(self, root):
        """
        将一棵二叉树的所有结点作为根节点进行序列化，记录该前序序列化字符串出现的次数。
        1、如果出现的次数大于1，那么就说明该序列重复出现。
        2、如果等于1，说明在这之前遇到过一次节点。
        最后统计完重复的后，返回结果；如果是空结点的话，返回一个任意非空字符串。

        时间空间复杂度O(n)
        """
        count = collections.Counter()
        ans = []
        def collect(node):
            if not node: return "#"
            serial = "{},{},{}".format(node.val, collect(node.left), collect(node.right))
            count[serial] += 1
            if count[serial] == 2:
                ans.append(node)
            return serial

        collect(root)
        return ans



class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

if __name__ == '__main__':
    root = build_tree2()
    sol = Solution()
    res = sol.findLeaves1(root)
    print(res)


    pass



