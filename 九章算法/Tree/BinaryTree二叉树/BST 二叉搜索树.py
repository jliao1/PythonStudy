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

    # lintcode Easy 1524 · Search in a Binary Search Tree
    def searchBST_recursive(self, root, val):
        if not root or root.val == val:
            # 要么是彻底找不到了，要么是已经找到了
            return root
        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    # lintcode Easy 1524 · Search in a Binary Search Tree
    def searchBST_iterative(self, root, val):
        while root and root.val != val:
            if val < root.val:
                root = root.left
            else:
                root = root.right
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

    def isValidBST4(self, root):
        return self.helper(root, float('-inf'), float('inf'))
    def dfs_for_isValidBST4(self, min_value, max_value):
        if not root:
            return True
        if root.val <= min_value:
            return False
        if root.val >= max_value:
            return False
        return self.dfs_for_isValidBST4(root.left, min_value, root.val) and self.helper(root.right, root.val, max_value)

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

           如果输入的array是[0,1,2,3,4]
           得到的tree就是   3
                         / \
                        0   3
                         \   \
                         1   4
        """
        return self.convert(A, 0, len(A) - 1)
    def convert(self, A, start, end):
        if start > end:
            return None

        # always choose left middle node as a root
        mid = (start + end) // 2  # //是整除的意思
        root = TreeNode(A[mid])
        root.left = self.convert(A, start, mid - 1)
        root.right = self.convert(A, mid + 1, end)
        return root

    # lintcode(力扣108) Easy 177也是1359 · Convert Sorted Array to Binary Search Tree With Minimal Height 力扣答案
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
        而这个 切片语句使速度变慢了，因为会生成新的数组，导致速度变慢
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

    # Lintcode hard 87 · Remove Node in Binary Search Tree  主要是 edge case太多，太烦了
    def removeNode1(self, root, value):
        """
        这是sheila教的方法，
        先找到 value 的 parent 和这个 value
        然后看要删除的 target node 是不是 叶子节点，或之有1个孩子，或有2个孩子，分这3种情况讨论
        写出来累死了…  我写得不太好的地方在于可能很多地方重复判断了？
        可以看下方法2 iterative 写法要简练点，把remove都整合到一个函数里了
        方法3 比 方法2 更简洁一些
        """
        dummy = TreeNode(-float('inf'))
        dummy.right = root
        target_node, parent = self.find_target_node_and_its_parent(dummy, value)

        # find nothing，doing nothing
        if not parent and not target_node:
            return root

        # target node is a leaf node
        if not target_node.left and not target_node.right:
            self.remove_leaf_node(parent, target_node)
        # target node only has one child
        elif target_node.left and not target_node.right:
            self.remove_node_with_one_child(parent, target_node)
        elif target_node.right and not target_node.left:
            self.remove_node_with_one_child(parent, target_node)
        # target node has 2 children
        elif target_node.right and target_node.left:
            self.remove_node_with_2_children(target_node)

        return dummy.right
    def remove_node_with_2_children(self, target):
        """
        策略是 find inorder predecessor
        从 target node 的位置开始，先go left once，然后 go as right as possible
        就可以找到 target 的 inorder predecessor
        然后把 inorder predecessor 位置的 value 和 target 位置的 value swap
        然后删掉 inorder predecessor 位置的 node
        """
        # go left once
        curr = target.left
        # go as right as possible
        curr_prev = target
        while curr.right:
            curr_prev = curr
            curr = curr.right
        # find target's inorder_predecessor
        inorder_predecessor = curr
        inorder_predecessor_parent = curr_prev
        # swap
        inorder_predecessor.val, target.val = target.val, inorder_predecessor.val
        # delete inorder_predecessor
        if inorder_predecessor.left:
            # inorder_predecessor has one child
            self.remove_node_with_one_child(inorder_predecessor_parent, inorder_predecessor)
        else:
            # inorder_predecessor has no child
            self.remove_leaf_node(inorder_predecessor_parent, inorder_predecessor)
    def remove_node_with_one_child(self, parent,  target):
        # 要删除的 target node 如果只有一个孩子，那么直接让 target 的 parent 指向 target 的唯一孩子
        if target.left:
            # 如果 target 唯一的孩子是左孩子
            target_child = target.left
        else:
            # 如果 target 唯一的孩子是右孩子
            target_child = target.right
        if parent.left and parent.left == target:
            # 如果 parent 的左孩子 是 target node
            parent.left = target_child
        else:
            # 如果 parent 的右孩子 是 target node
            parent.right = target_child
    def remove_leaf_node(self, parent, target):
        # 要删除的 target node 如果是 叶子节点，那么直接让 target 的 parent 指向 None
        if parent.left and parent.left == target:
            # 如果 target node 是 parent 的左孩子
            parent.left = None
        else:
            # 那么 target node 是 parent 的左孩子
            parent.right = None
    def find_target_node_and_its_parent(self, root, value):
        if not root:
            return None

        if root.left:
            if root.left.val == value:
                return root.left, root
            elif value < root.val:
                return self.find_target_node_and_its_parent(root.left, value)

        if root.right:
            if root.right.val == value:
                return root.right, root
            elif value >= root.val:
                return self.find_target_node_and_its_parent(root.right, value)

        return None, None

    # Lintcode hard 87 · Remove Node in Binary Search Tree 这个思路和方法1一样，但是iterative写法，写得比方法1简练点
    def removeNode2(self, root, value):
        # # 这个部分是处理null case 不过其实以下这2句不写也是可以的，因为 check parent of the node to delete
        # #            时已经处理当parent是None的情况了（也就是cur指向root，但由于root是None导致parent也是None根本没移动）
        # if root is None:
        #     return root

        # 这个部分是find the node to delete and its parent
        parent, target = None, root
        # 找cur把它指向我们想要的value node
        # 循环停止条件是：找到 target.val 就是 value 停止
        #              或者 target 是 None 为止 (cur如果是None了说明tree里根本不存在value)
        # 但有2种情况根本就不会进入while循环：
        #         (1) 但如果 root是None ，那么target一开始也就是None
        #         (2) value 就在 root
        #         这两种情况，由于根本不进入while循环，那么parent = None
        while target and target.val != value:
            parent = target
            if target.val > value:
                target = target.left
            else:
                target = target.right

        # 这部分开始 remove 了，remove()返回的其实都是删除了target后上位的节点，正好可以连到 parent 下，
        # 其实 check parent是不是None, 就是在处理 root 是 null case 的情况（就是最开始2行代码）
        #                            或者 value 就在 root 为止找到的情况，删除 target = root 根节点
        if parent is None:
            return self.remove_helper(target)
        # 上面说了，如果tree里没找到value，cur会是None，parent.left就是None，就相当于调用self.remove(None)，remove函数也只会返回个None，不会对结果有啥影响
        # 如果找到value了，要么 parent左孩子是 target value node，要么右孩子是，就删除它
        elif parent.left is target:
            parent.left = self.remove_helper(target)
        else:
            parent.right = self.remove_helper(target)

        return root
    def remove_helper(self, target):
        # case1：target is null: 就是上面说的 根本没找到value的情况，target就会是None，再返回target相当于啥也没干
        if target is None:
            return target

        # case2: if target is leaf
        if target.left is None and target.right is None:
            return None   # 返回 None 因为是让 parent 的左孩子/右孩子 连上 None

        # case3: if target has only one child
        if target.left is None:  # 如果左孩子不存在，那就返回右孩子，好让parent上target的右孩子
            return target.right
        if target.right is None:  # 如果右孩子不存在，那就返回左孩子，好让parent上target的左孩子
            return target.left

        # case4：if target has 2 children, find max node in left subtree（target 的 inorder predecessor）
        prev, cur = None, target.left  # 因为反正target也是有2孩子的，cur先向target的左走1次
        while cur.right:               # 然后 cur go as right as possible
            prev = cur
            cur = cur.right
        # 走到这里说明 cur 已指向了 inorder predecessor

        # make inorder predecessor as new root (move the pointer not copy value)
        # 这步好难想……
        cur.right = target.right
        if target.left is not cur:  # 如果这条语句是True，说明cur除了往左target左走了一步后还向右下继续走了几步
            prev.right = cur.left
            cur.left = target.left
        return cur

    # Lintcode hard 87 · Remove Node in Binary Search Tree 很棒的分治法思路！！也是解这题最简单的方法
    def removeNode3(self, root, value):
        """
        这个方法3其实比方法1和2更简练些
        recursion去找要删除的node，找到后直接进行删除操作，再返回连接上原parent
        O(h) time, O(h) space, h: height of tree
        """
        # null case
        if root is None:
            return root

        # check if node to delete is in left/right subtree
        if value < root.val:
            root.left = self.removeNode3(root.left, value)
        elif value > root.val:
            root.right = self.removeNode3(root.right, value)
        else:  # 说明 root.val == value 找到了！就开始做删除root的操作了！
            # if root is has 2 children
            if root.left and root.right:
                left_max = self.find_Max_of_left_tree(root)
                # 把 left_max.val 的值 给 root.val 的位置
                root.val = left_max.val
                # 然后从 root 的 left tree 进入去删掉 这个 left_max
                root.left = self.removeNode3(root.left, left_max.val)
            # if root has only one child
            elif root.left:
                root = root.left
            elif root.right:
                root = root.right
            # if root has no child, it's leaf node
            else:
                root = None

        return root
    # find max node in left subtree of root，就是找删除target的 inorder_predecessor
    def find_Max_of_left_tree(self, target):
        # 反正target也有2个孩子，向左移动1次
        node = target.left
        # 然后go right as possible
        while node.right:
            node = node.right
        return node

    # Lintcode hard 87 · Remove Node in Binary Search Tree
    def removeNode4(self, root, value):
        """
        这个思路是对原列表左一次inorder traversal 但是要删除 value 后生成一个 list
        然后依据这个list 重建一个(高度尽量小的) 二叉树
        但这种可能不太符合题目说的，因为题目要求是，没找到value的话就doing nothing
        """
        self.ans = []
        self.inorder_traversal_list(root, value)
        return self.build_tree_from_inorder_traversal(0, len(self.ans) - 1)
    def inorder_traversal_list(self, root, value):
        if root is None:
            return

        self.inorder_traversal_list(root.left, value)
        if root.val != value:
            self.ans.append(root.val)
        self.inorder_traversal_list(root.right, value)
    def build_tree_from_inorder_traversal(self, l, r):
        if l == r:
            node = TreeNode(self.ans[l])
            return node

        if l > r:
            return None

        mid = (l + r) // 2
        node = TreeNode(self.ans[mid])
        node.left = self.build_tree_from_inorder_traversal(l, mid - 1)
        node.right = self.build_tree_from_inorder_traversal(mid + 1, r)
        return node

    # Lintcode(力扣669) Medium 701 · Trim a Binary Search Tree 分治法思路！
    def trimBST(self, root, minimum, maximum):
        """
        时间O(n)因为在最坏情况下会visit evry node once time，好的情况下可以直接仍掉某一拖子树不需要访问就可以小于O(n)
        空间复杂度O(h)
        """
        if not root:
            return None

        curr_val = root.val

        if curr_val < minimum:
            # 情况1: root node 要删掉的，它的左子树也不用看了，只往它右子树看
            root = self.trimBST(root.right, minimum, maximum)
        elif maximum < curr_val:
            # 情况2: root node 要删掉的，它的右子树也不用看了，只往它左子树看
            root = self.trimBST(root.left, minimum, maximum)
        else:
            # 情况3: minimum ≤ curr_val ≤ maximum
            # root node 要保留，它左右子树都要看一下
            root.left = self.trimBST(root.left, minimum, root.val)
            root.right = self.trimBST(root.right, root.val, maximum)

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
    array = [0,1,2,3,4]
    sol = Solution()
    res = sol.sortedArrayToBST1(array)
    print(sol.cnt)


