import copy
from collections import deque
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


class Solution:
    def __init__(self):
        self.counter = 0
        self.string = []   # lintcode(力扣606) Easy 1137 · Construct String from Binary Tree

    # lintcode(力扣94) Easy 67 · Binary Tree Inorder Traversal  这道题挑战让你用iterative写法
    def inorderTraversal_iterative(self, root):
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
        空间O(h)
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



if __name__ == '__main__':
    # root = build_tree1()
    # res = postorder_traverse(root)

    root1 = build_tree1()

    sol = Solution()
    res = sol.rightSideView(root1)
    print(res)
    print(sol.counter)


    pass



