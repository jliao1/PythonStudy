class TreeNode:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# BST 性质简介 时空复杂度
"""
性质：
（1）对BST进行中序遍历 left-root-right 得到得是一个递增(非降)序列
（2）有插入操作（普通的二叉树binary tree没有(因为普通的结构太松散混乱了)，普通的只有深度/宽度优先遍历）
（3）查找高校
（4）常用操作: 插入，查找，删除（用得少）

查找(插入)操作的 时间、空间 复杂度
时间复杂度,是O(h): 每次操作都会往下走一层，每次走的都是BST的一条路径，深度是多少时间消耗就是多少
空间复杂度,是O(h)：递归的时候，消耗了栈空间。那递归多少层呢？
（二分搜索的思想就是每次扔掉一半儿）

树层数(高度)h与节点数n的关系：
worst case：N（退化为线性）
best case：logN （平衡的BST）

平衡BST应用是，比如你访问一个元素，你想知道之前有没有访问过，每次访问完就把它插到BST里，
下次访问再看它是不是已经存在BST里，是否contains
所以BST是个很适合于做记录的数据结构，因为插入查找特别快
"""

# BST 的构建，添加，查找
class BST:

    def __init__(self):
        self.__root = None  # 存根节点代表整棵树，初始状态下是None
        # __两个下划线代表这个变量是private的

    # add在很多地方后，其实也会保持是个BST，那么add在哪里呢？尽量add在叶子上(就不用把树结构拆开重连了),而且,一定可以add在叶子上的！
    def add(self, val):
        self.__root = self.__add_helper(self.__root, val)
        #  这里需要 等号赋值，最开始root是 None，插入后要跟root连接起来，才不会是None
    # 这个 helper 是要返回一个值的，什么值呢？就是input进入去的root
    def __add_helper(self, root, val):
        if not root:                                     # 如果root压根儿就不存在，说明这个树没有
            return TreeNode(val)                         # 那要add的这个val就作为这个树唯一的node返回了
        # 如果传入的root存在，就要开始判断了
        if val < root.val:
            root.left = self.__add_helper(root.left, val)
            # 为什么要连接一下呢？因为有些root.left是None，新插入的节点要返回连接上root.left 这样root.left就不再是None了
            #                   有些root.left不是None，这里 = 连接一下 也还是与原来得节点相连，没啥意义，但是也没副作用。既然有的情况需要连，有的情况不需要连，那就都得连接
            # 这个 = 赋值操作在插入的最后一层才有意义，因为最后一层的节点的left原来是None,需要接上新节点才行
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


class Solution:
    def __init__(self):
        self.cnt = 0

    # lintcode Easy 85 · Insert Node in a Binary Search Tree 这是 recursive写法
    def insertNode1(self, root, node):
        if root is None:
            return node
        if root.val > node.val:
            root.left = self.insertNode1(root.left, node)
        else:
            root.right = self.insertNode1(root.right, node)
        return root

    # lintcode Easy 85 · Insert Node in a Binary Search Tree 这是 iterative写法
    def insertNode2(self, root, node):
        if root is None:
            return node

        cur = root
        while cur != node:
            if node.val < cur.val:
                # 找到叶子节点就插入
                if cur.left is None:  # 说明cur是叶子节点了
                    cur.left = node
                # 更新 cur
                cur = cur.left
            else:
                # 找到叶子节点就插入
                if cur.right is None:  # 说明cur是叶子节点了
                    cur.right = node
                # 更新 cur
                cur = cur.right
        return root

    # lintcode(力扣98) Medium 95 · Validate Binary Search Tree
    def isValidBST1(self, root):
        """
        思路是：对它进行中序遍历，结果存在list里，只要list里的元素，不递减(最好递增)
        时间空间复杂度O(n)
        但我们真的一定要记录 inorder tranverse 的 list？其实没有必要，可以看下解法2
        """
        # 中序遍历结果要存在一个列表里
        self.res = []   # 这个成员变量 res 可以定义在这里，供本class下其他函数直接调用。但写在这里不规范，最好还是写在 init 里

        self.dfsForIsValidBST1(root)

        # 判断 res 里是不是严格单调递增
        for i in range(1, len(self.res)):  # 判断范围从1到res的最高位就行. why 1? 因为1处跟前一位比较就行
            if self.res[i] <= self.res[i - 1]:
                return False

        return True
    def dfsForIsValidBST1(self, root):
        if not root:
            return
        self.dfsForIsValidBST1(root.left)
        self.res.append(root.val)          # 在这里 append 不需要重新 extend在 res 上了，体会下
        self.dfsForIsValidBST1(root.right)

    # lintcode(力扣98) Medium 95 · Validate Binary Search Tree
    def isValidBST2(self, root: TreeNode) -> bool:
        """
        worst case下才是O(n)，
        这是我自己写的，run得比较慢，因为 但是 dfsForIsValidBST2()里 if 语句太多了，导致 O(n)的系数变大，最终运行时间变长
        方法3的运行时间还行，但……感觉又不readble了，反正可以都看看
        """
        self.prev = -float('inf')
        isValid = self.dfsForIsValidBST2(root)
        return isValid
    def dfsForIsValidBST2(self, root):

        if not root:
            return True

        isLeftOK = self.dfsForIsValidBST2(root.left)

        isRootOK = True if self.prev < root.val else False

        # update self.prev
        self.prev = root.val

        if isLeftOK and isRootOK:

            isRightOK = self.dfsForIsValidBST2(root.right)

        else:
            return False

        return isRightOK

    # lintcode(力扣98) Medium 95 · Validate Binary Search Tree
    def isValidBST3(self, root):
        """
        这种recurssive写法也不错，但感觉还是版本1更readable
        """
        def inorder(root):
            if not root:
                return True

            # 访问左孩子：如果左孩子不满足，return False
            if not inorder(root.left):
                return False

            # 访问中间：本该小于的，但如果大于了就返回 false
            if self.prev >= root.val:
                return False
            # 能走到这里说明目前正常，update一下prev变成当前的root value
            self.prev = root.val

            # 访问右孩子：
            return inorder(root.right)

        # 这个主要是用来记录前一个点的value
        self.prev = -float('inf')
        return inorder(root)

    # leetcode Medium 98. Validate Binary Search Tree
    def isValidBSTWrongAnswer(self, root):
        """
        这个是sheila教的错误答案
        这个写法，只能保证 每一个节点，left child < 自己 < right child
        而BST 应该是 所有的 left children 都 < 自己 < 所有的 right children
        """
        if not root:
            return True

        if root.left and root.left.val >= root.val or root.right and root.right.val <= root.val:
            return False

        a = self.isValidBST4(root.left)
        b = self.isValidBST4(root.right)
        return a and b

    # lintcode(力扣108) Easy 177 · Convert Sorted Array to Binary Search Tree With Minimal Height 九章答案
    def sortedArrayToBST1(self, A):
        """
        1. 一个sorted array虽然是一个BST的中序遍历结果，但是它不能推出唯一的一个BST结构
           意思是 "sorted array -> BST" has multiple solutions
        2. 为了高度尽量小，我们这个tree得要balanced
           （the depths of the two subtrees of every node never differ by more than 1）
           但这也不能使得tree唯一
        3. the height-balanced restriction means that
           at each step one has to pick up the number in the middle as a root
           That works fine with arrays containing odd number of elements
           but there is no predefined choice for arrays with even number of elements
           One could choose left middle element, or right middle one,
           and both choices will lead to different height-balanced BSTs,
           both solutions will be accepted
        """
        return self.convert(A, 0, len(A) - 1)
    def convert(self, A, start, end):
        if start > end:
            return None
        if start == end:
            return TreeNode(A[start])

        # always choose left middle node as a root
        mid = (start + end) // 2
        root = TreeNode(A[mid])
        root.left = self.convert(A, start, mid - 1)
        root.right = self.convert(A, mid + 1, end)
        return root

    # lintcode(力扣108) Easy 177 · Convert Sorted Array to Binary Search Tree 力扣答案
    def sortedArrayToBST2(self, A):
        """
        这种写法思路与方法1类似，也是 in-place的，只不过把 递归函数写在 主函数里啦
        Time complexity: O(N) since we visit each node exactly once.
        Space complexity: O(N). O(N) to keep the output, and O(logN) for the recursion stack.
        """
        def helper(left, right):
            if left > right:
                return None

            # always choose left middle node as a root
            p = (left + right) // 2
            '''
            # 如果选择右边就这么写 always choose right middle node as a root
            p = (left + right) // 2 
            if (left + right) % 2:
                p += 1            
            '''

            # preorder traversal: node -> left -> right
            root = TreeNode(A[p])
            root.left = helper(left, p - 1)
            root.right = helper(p + 1, right)
            return root

        return helper(0, len(A) - 1)


    # lintcode(力扣108) Easy 177 · Convert Sorted Array to Binary Search Tree With Minimal Height
    def sortedArrayToBST3(self, A):
        """
        这个方法比方法1和2要慢，因为1和2是in-place的
        而这个 切片变慢了，因为会生成新的数组，导致速度变慢
        """
        return self.converse(A)
    def converse(self, B):
        if not B:
            return None
        mid = (len(B) - 1) // 2
        root = TreeNode(B[mid])
        root.left = self.converse(B[0:mid])  # 切片变慢了
        root.right = self.converse(B[mid+1:len(B)])
        return root


def build_tree():
    node_1 = TreeNode(5)
    node_2 = TreeNode(4)
    node_3 = TreeNode(6)
    node_4 = TreeNode(3)
    node_5 = TreeNode(7)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.left = node_2
    node_1.right = node_3


    node_3.left = node_4

    node_3.right = node_5
    #
    # node_5.left = node_7
    # node_5.right = node_8
    #
    # node_6.left = node_9

    return node_1

if __name__ == '__main__':

    root = build_tree()
    sol = Solution()
    res = sol.isValidBST1(root)
    print(sol.cnt)


